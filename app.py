import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import json
from scipy import stats
import requests
from io import StringIO

# Mapeamento de classes para descrições
CLASS_MAPPING = {
    0: "Baixo Desempenho",
    1: "Médio Desempenho",
    2: "Alto Desempenho"
}

# Definir cores personalizadas para o tema
THEME_COLORS = {
    'background': 'rgba(0,0,0,0)',
    'text': '#2d3748',  # Cinza escuro
    'grid': 'rgba(45,55,72,0.1)',  # Cinza escuro com transparência
    'marker_colors': ['#ef4444', '#3b82f6', '#22c55e']  # Vermelho, Azul, Verde
}

def format_large_number(number):
    """Formata números grandes para notação mais legível (K, MI, BI, TRI)"""
    abs_num = abs(number)
    sign = '-' if number < 0 else ''
    
    if abs_num >= 1e12:
        return f"{sign}{abs_num/1e12:.2f}TRI"
    elif abs_num >= 1e9:
        return f"{sign}{abs_num/1e9:.2f}BI"
    elif abs_num >= 1e6:
        return f"{sign}{abs_num/1e6:.2f}MI"
    elif abs_num >= 1e3:
        return f"{sign}{abs_num/1e3:.2f}K"
    else:
        return f"{sign}{abs_num:.2f}"

def prepare_plot_data(df):
    """Prepara os dados para visualização aplicando transformação log1p nas métricas relevantes"""
    plot_data = df.copy()
    
    # Lista de colunas para aplicar log1p
    log_columns = [
        'marketcap', 
        'revenue_ttm', 
        'earnings_ttm', 
        'price',
        'pe_ratio_ttm',
        'dividend_yield_ttm'
    ]
    
    # Aplicar transformação log1p
    for col in log_columns:
        # Para valores negativos, preservar o sinal após a transformação para eixos x e y
        plot_data[f'{col}_log'] = np.sign(plot_data[col]) * np.log1p(np.abs(plot_data[col]))
    
    # Criar coluna especial para tamanho dos pontos (sempre positiva)
    plot_data['size_value'] = np.log1p(np.abs(plot_data['revenue_ttm']))
    
    # Normalizar tamanho dos pontos para um intervalo razoável (10-50)
    min_size = plot_data['size_value'].min()
    max_size = plot_data['size_value'].max()
    plot_data['size_value'] = 10 + 40 * (plot_data['size_value'] - min_size) / (max_size - min_size)
    
    # Adicionar informações para o hover com valores formatados
    plot_data['hover_text'] = plot_data.apply(
        lambda x: f"Company: {x['name']}<br>" +
                 f"Country: {x['country']}<br>" +
                 f"Market Cap: {format_large_number(x['marketcap'])}<br>" +
                 f"P/E Ratio: {format_large_number(x['pe_ratio_ttm'])}<br>" +
                 f"Revenue: {format_large_number(x['revenue_ttm'])}<br>" +
                 f"Performance: {CLASS_MAPPING[x['pc_class']]}",
        axis=1
    )
    
    return plot_data

# Configuração da página
st.set_page_config(
    page_title="Stock Market Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Forçar tema light
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        .stApp {
            background-color: #ffffff;
        }
        .main {
            background-color: #f8f9fa;
        }
    </style>
""", unsafe_allow_html=True)

# Função para carregar ou treinar o modelo
@st.cache_resource
def load_or_create_model():
    try:
        # URL do modelo no GitHub
        model_url = "https://raw.githubusercontent.com/sidnei-almeida/tcc_streamlit/refs/heads/main/Random_Forest_model.joblib"
        
        # Fazer requisição GET para o arquivo
        response = requests.get(model_url)
        response.raise_for_status()  # Levanta exceção para erros HTTP
        
        # Salvar o modelo temporariamente e carregar
        with open('temp_model.joblib', 'wb') as f:
            f.write(response.content)
        
        model = joblib.load('temp_model.joblib')
        
        # Verificar se o modelo está prevendo corretamente
        data = load_data()
        if data is not None:
            X = data.drop(['name', 'country', 'pc_class'], axis=1)
            y = data['pc_class']
            
            # Fazer algumas previsões de teste
            test_preds = model.predict(X[:10])
            
            # Se todas as previsões forem a mesma classe, há algo errado
            if len(set(test_preds)) == 1:
                st.warning("Modelo carregado pode estar com problemas. Treinando novo modelo...")
                raise Exception("Modelo com previsões uniformes")
        
        return model
        
    except Exception as e:
        st.info("Treinando novo modelo Random Forest...")
        # Se não conseguir carregar, criar e treinar um novo modelo
        data = load_data()
        if data is None:
            st.error("Não foi possível carregar os dados para treinar o modelo.")
            return None
            
        X = data.drop(['name', 'country', 'pc_class'], axis=1)
        y = data['pc_class']
        
        # Criar e treinar o modelo
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Treinar o modelo
        model.fit(X, y)
        
        # Verificar distribuição das previsões
        train_preds = model.predict(X)
        class_dist = pd.Series(train_preds).value_counts()
        st.write("Distribuição das classes nas previsões de treino:", class_dist)
        
        # Salvar o modelo localmente
        joblib.dump(model, 'temp_model.joblib')
        
        return model

# Função para carregar os dados
@st.cache_data
def load_data():
    try:
        # URL do dataset no GitHub
        url = "https://raw.githubusercontent.com/sidnei-almeida/tcc_streamlit/refs/heads/main/data.csv"
        
        # Fazer requisição GET para o arquivo
        response = requests.get(url)
        response.raise_for_status()  # Levanta exceção para erros HTTP
        
        # Ler o CSV da resposta
        data = pd.read_csv(StringIO(response.text))
        
        if data.empty:
            st.error("O arquivo de dados está vazio.")
            return None
        
        # Verificar colunas necessárias
        required_columns = [
            'name', 'country', 'pc_class',
            'dividend_yield_ttm', 'earnings_ttm', 'marketcap',
            'pe_ratio_ttm', 'revenue_ttm', 'price',
            'gdp_per_capita_usd', 'gdp_growth_percent', 'inflation_percent',
            'interest_rate_percent', 'unemployment_rate_percent', 'exchange_rate_to_usd'
        ]
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            st.error(f"Colunas ausentes no arquivo de dados: {', '.join(missing_columns)}")
            return None
            
        return data
        
    except requests.RequestException as e:
        st.error(f"Erro ao carregar os dados do GitHub: {str(e)}")
        return None
    except pd.errors.EmptyDataError:
        st.error("O arquivo de dados está vazio.")
        return None
    except Exception as e:
        st.error(f"Erro ao processar os dados: {str(e)}")
        return None

# Função para remover outliers usando IQR
def remove_outliers_iqr(df):
    df_clean = df.copy()
    
    # Identificar colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Calcular Q1, Q3 e IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir limites
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Substituir outliers pela mediana
        median_val = df[col].median()
        df_clean.loc[df_clean[col] < lower_bound, col] = median_val
        df_clean.loc[df_clean[col] > upper_bound, col] = median_val
    
    return df_clean

# Função para preparar os dados para predição com pré-processamento completo
def prepare_data_complete(df, is_training_data=True):
    """
    Prepara os dados para o modelo, aplicando o mesmo pré-processamento dos dados de treino
    """
    # Separar features numéricas
    if is_training_data:
        features_df = df.drop(['name', 'country', 'pc_class'], axis=1, errors='ignore')
    else:
        features_df = df.drop(['name', 'country'], axis=1, errors='ignore')
    
    # Remover outliers
    features_df_clean = remove_outliers_iqr(features_df)
    
    # Padronização
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df_clean)
    
    return features_scaled, scaler

# Função para fazer previsões em lote
def batch_predict(df):
    try:
        # Preparar dados de entrada
        input_data = df.copy()
        if 'name' in input_data.columns and 'country' in input_data.columns:
            features_df = input_data.drop(['name', 'country'], axis=1, errors='ignore')
        else:
            features_df = input_data
        
        # Verificar se as colunas estão na mesma ordem dos dados de treino
        train_data = load_data()
        train_features = train_data.drop(['name', 'country', 'pc_class'], axis=1).columns
        features_df = features_df[train_features]  # Reordenar colunas para match com treino
        
        # Fazer previsões diretamente com os dados fornecidos
        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)
        
        # Verificar distribuição das previsões
        pred_dist = pd.Series(predictions).value_counts()
        st.write("Distribuição das previsões:", pred_dist)
        
        # Adicionar previsões ao dataframe original
        result_df = df.copy()
        result_df['pc_class'] = predictions
        result_df['pc_class_desc'] = result_df['pc_class'].map(CLASS_MAPPING)
        
        # Adicionar probabilidades
        for i in range(3):
            result_df[f'prob_class_{i}'] = probabilities[:, i]
            result_df[f'prob_{CLASS_MAPPING[i]}'] = probabilities[:, i]
        
        # Adicionar versões log das métricas para visualização
        result_df = prepare_plot_data(result_df)
        
        return result_df
        
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        return None

# Carregar modelo e dados
model = load_or_create_model()
data = load_data()

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2b6cb0;
        color: white;
    }
    .stSelectbox {
        color: #2b6cb0;
    }
    .plot-container {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("📊 Stock Market Analysis & Prediction")
st.markdown("---")

# Carregar dados
data = load_data()

# Layout principal com três colunas
col1, col2, col3 = st.columns(3)

# Métricas principais
with col1:
    st.metric(
        "Total Companies",
        len(data)
    )

with col2:
    avg_market_cap = data['marketcap'].mean()
    st.metric(
        "Average Market Cap",
        format_large_number(avg_market_cap)
    )

with col3:
    avg_pe = data['pe_ratio_ttm'].mean()
    st.metric(
        "Average P/E Ratio",
        format_large_number(avg_pe)
    )

st.markdown("---")

# Tabs para diferentes visualizações
tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Analysis", "🎯 Performance Metrics", "🤖 Prediction Tool", "📊 Batch Prediction"])

with tab1:
    # Preparar dados para visualização
    plot_data = prepare_plot_data(data)
    
    # Gráfico de dispersão interativo
    fig = px.scatter(
        plot_data,
        x='marketcap_log',
        y='pe_ratio_ttm_log',
        size='size_value',  # Usar a nova coluna normalizada para tamanho
        color='pc_class',
        hover_name='name',
        hover_data={
            'marketcap_log': False,
            'pe_ratio_ttm_log': False,
            'size_value': False,
            'pc_class': False,
            'hover_text': True
        },
        color_discrete_sequence=THEME_COLORS['marker_colors'],
        title='Market Cap vs P/E Ratio por Classe de Desempenho (Escala Log)'
    )
    
    # Atualizar layout do gráfico
    fig.update_layout(
        template=None,  # Remove o template padrão
        plot_bgcolor=THEME_COLORS['background'],
        paper_bgcolor=THEME_COLORS['background'],
        title=dict(
            font=dict(size=24, color=THEME_COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            title=dict(text='Classe de Desempenho', font=dict(color=THEME_COLORS['text'])),
            font=dict(color=THEME_COLORS['text']),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=THEME_COLORS['text'],
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial"
        ),
        xaxis=dict(
            title='Market Cap (log scale)',
            title_font=dict(size=14, color=THEME_COLORS['text']),
            tickfont=dict(size=12, color=THEME_COLORS['text']),
            gridcolor=THEME_COLORS['grid'],
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='P/E Ratio (log scale)',
            title_font=dict(size=14, color=THEME_COLORS['text']),
            tickfont=dict(size=12, color=THEME_COLORS['text']),
            gridcolor=THEME_COLORS['grid'],
            showgrid=True,
            zeroline=False
        )
    )
    
    # Atualizar os marcadores
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color=THEME_COLORS['text'])
        ),
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Distribuição de classes por país
    country_class_dist = pd.crosstab(data['country'], data['pc_class'])
    country_class_dist.columns = ['Baixo Desempenho', 'Médio Desempenho', 'Alto Desempenho']
    
    fig_dist = px.bar(
        country_class_dist,
        title='Distribuição de Classes de Desempenho por País',
        color_discrete_sequence=THEME_COLORS['marker_colors'],
        barmode='group'
    )
    
    fig_dist.update_layout(
        template=None,
        plot_bgcolor=THEME_COLORS['background'],
        paper_bgcolor=THEME_COLORS['background'],
        title=dict(
            font=dict(size=24, color=THEME_COLORS['text']),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            title=dict(text='Classe de Desempenho', font=dict(color=THEME_COLORS['text'])),
            font=dict(color=THEME_COLORS['text']),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=THEME_COLORS['text'],
            borderwidth=1
        ),
        xaxis=dict(
            title='País',
            title_font=dict(size=14, color=THEME_COLORS['text']),
            tickfont=dict(size=12, color=THEME_COLORS['text']),
            tickangle=-45,
            gridcolor=THEME_COLORS['grid'],
            showgrid=False
        ),
        yaxis=dict(
            title='Número de Empresas',
            title_font=dict(size=14, color=THEME_COLORS['text']),
            tickfont=dict(size=12, color=THEME_COLORS['text']),
            gridcolor=THEME_COLORS['grid'],
            showgrid=True
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot com valores transformados
        metrics_to_plot = ['dividend_yield_ttm_log', 'pe_ratio_ttm_log', 'price_log']
        metrics_labels = {
            'dividend_yield_ttm_log': 'Dividend Yield (Log)',
            'pe_ratio_ttm_log': 'P/E Ratio (Log)',
            'price_log': 'Price (Log)'
        }
        
        fig_box = px.box(
            plot_data,
            y=metrics_to_plot,
            color='pc_class',
            title='Métricas por Classe (Log)',
            color_discrete_sequence=THEME_COLORS['marker_colors'],
            labels=metrics_labels
        )
        
        fig_box.update_layout(
            template=None,
            plot_bgcolor=THEME_COLORS['background'],
            paper_bgcolor=THEME_COLORS['background'],
            title=dict(
                font=dict(size=16, color=THEME_COLORS['text']),
                x=0.5,
                xanchor='center',
                y=0.95  # Ajuste da posição vertical do título
            ),
            legend=dict(
                title=dict(text='Classe de Desempenho', font=dict(color=THEME_COLORS['text'], size=12)),
                font=dict(color=THEME_COLORS['text'], size=10),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=THEME_COLORS['text'],
                borderwidth=1
            ),
            xaxis=dict(
                title_font=dict(size=12, color=THEME_COLORS['text']),
                tickfont=dict(size=10, color=THEME_COLORS['text']),
                gridcolor=THEME_COLORS['grid']
            ),
            yaxis=dict(
                title_font=dict(size=12, color=THEME_COLORS['text']),
                tickfont=dict(size=10, color=THEME_COLORS['text']),
                gridcolor=THEME_COLORS['grid'],
                showgrid=True
            ),
            margin=dict(t=50)  # Reduzir margem superior
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Matriz de correlação com valores transformados
        numeric_cols_log = [
            'dividend_yield_ttm_log', 
            'earnings_ttm_log', 
            'marketcap_log', 
            'pe_ratio_ttm_log',
            'revenue_ttm_log', 
            'price_log'
        ]
        
        # Criar labels mais legíveis para a matriz de correlação
        corr_labels = {
            'dividend_yield_ttm_log': 'Div. Yield',
            'earnings_ttm_log': 'Earnings',
            'marketcap_log': 'Market Cap',
            'pe_ratio_ttm_log': 'P/E Ratio',
            'revenue_ttm_log': 'Revenue',
            'price_log': 'Price'
        }
        
        corr_matrix = plot_data[numeric_cols_log].corr()
        
        # Criar anotações com os valores formatados
        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, value in enumerate(row):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f'{value:.2f}',
                        showarrow=False,
                        font=dict(
                            size=10,
                            color='black' if abs(value) < 0.7 else 'white'
                        )
                    )
                )
        
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu',
            title='Correlações (Log)',
            labels=dict(x='', y=''),
            x=list(corr_labels.values()),
            y=list(corr_labels.values())
        )
        
        fig_corr.update_layout(
            template=None,
            plot_bgcolor=THEME_COLORS['background'],
            paper_bgcolor=THEME_COLORS['background'],
            title=dict(
                font=dict(size=16, color=THEME_COLORS['text']),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            xaxis=dict(
                tickfont=dict(size=10, color=THEME_COLORS['text']),
                tickangle=45
            ),
            yaxis=dict(
                tickfont=dict(size=10, color=THEME_COLORS['text'])
            ),
            margin=dict(t=50),
            annotations=annotations  # Adicionar as anotações com os valores
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.header("Stock Performance Predictor")
    st.markdown("""
    Use this tool to predict the performance class of a stock based on its financial metrics.
    Enter the values below and click 'Predict' to see the results.
    
    **Performance Classes:**
    - 0: Baixo Desempenho
    - 1: Médio Desempenho
    - 2: Alto Desempenho
    """)

    # Formulário para entrada de dados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dividend_yield = st.number_input('Dividend Yield (TTM)', value=0.0)
        earnings = st.number_input('Earnings (TTM)', value=0.0)
        marketcap = st.number_input('Market Cap', value=0.0)
        
    with col2:
        pe_ratio = st.number_input('P/E Ratio (TTM)', value=0.0)
        revenue = st.number_input('Revenue (TTM)', value=0.0)
        price = st.number_input('Price', value=0.0)
        
    with col3:
        gdp = st.number_input('GDP per Capita (USD)', value=0.0)
        gdp_growth = st.number_input('GDP Growth (%)', value=0.0)
        inflation = st.number_input('Inflation (%)', value=0.0)

    # Botão de predição
    if st.button('Predict Performance Class'):
        # Preparar dados de entrada
        input_data = pd.DataFrame({
            'dividend_yield_ttm': [dividend_yield],
            'earnings_ttm': [earnings],
            'marketcap': [marketcap],
            'pe_ratio_ttm': [pe_ratio],
            'revenue_ttm': [revenue],
            'price': [price],
            'gdp_per_capita_usd': [gdp],
            'gdp_growth_percent': [gdp_growth],
            'inflation_percent': [inflation],
            'interest_rate_percent': [0],  # placeholder
            'unemployment_rate_percent': [0],  # placeholder
            'exchange_rate_to_usd': [1],  # placeholder
            'inflation': [-inflation],
            'interest_rate': [0],
            'unemployment': [0]
        })
        
        # Escalar dados
        _, scaler = prepare_data_complete(data)
        input_scaled = scaler.transform(input_data)
        
        # Fazer predição
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # Exibir resultados
        predicted_class = prediction[0]
        st.success(f'Predicted Performance Class: {predicted_class} ({CLASS_MAPPING[predicted_class]})')
        
        # Gráfico de probabilidades
        fig_proba = go.Figure(data=[
            go.Bar(
                x=['Baixo Desempenho', 'Médio Desempenho', 'Alto Desempenho'],
                y=prediction_proba[0],
                marker_color=['#ef4444', '#3b82f6', '#22c55e']
            )
        ])
        fig_proba.update_layout(
            title='Prediction Probabilities',
            yaxis_title='Probability',
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_proba, use_container_width=True)

with tab4:
    st.header("Batch Prediction")
    st.markdown("""
    Upload a CSV file with the same features as the training data (excluding pc_class) to get predictions for multiple stocks at once.
    
    **Required columns:**
    ```
    - dividend_yield_ttm
    - earnings_ttm
    - marketcap
    - pe_ratio_ttm
    - revenue_ttm
    - price
    - gdp_per_capita_usd
    - gdp_growth_percent
    - inflation_percent
    - interest_rate_percent
    - unemployment_rate_percent
    - exchange_rate_to_usd
    ```
    
    **Optional columns:**
    - name
    - country
    
    **Note:** The input data should already be preprocessed and scaled in the same way as the training data.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Carregar dados do arquivo
            input_df = pd.read_csv(uploaded_file)
            
            # Verificar colunas necessárias
            required_cols = set(data.drop(['pc_class', 'name', 'country'], axis=1).columns)
            missing_cols = required_cols - set(input_df.columns)
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Mostrar preview dos dados
                st.subheader("Data Preview")
                st.dataframe(input_df.head())
                
                # Botões lado a lado
                col1, col2 = st.columns(2)
                
                with col1:
                    predict_button = st.button("Make Predictions", type="primary")
                
                # Variável de sessão para armazenar resultados
                if predict_button:
                    # Fazer previsões com pré-processamento
                    with st.spinner('Processing data and generating predictions...'):
                        result_df = batch_predict(input_df)
                    
                    if result_df is not None:
                        # Armazenar resultados na sessão
                        st.session_state['prediction_results'] = result_df
                        
                        # Exibir resultados
                        st.success("Predictions completed successfully!")
                        
                        # Mostrar distribuição das previsões
                        fig_dist = px.pie(
                            result_df,
                            names='pc_class_desc',
                            title='Distribution of Predicted Classes',
                            color_discrete_sequence=THEME_COLORS['marker_colors']
                        )
                        
                        # Atualizar layout do gráfico de pizza
                        fig_dist.update_layout(
                            title=dict(
                                font=dict(size=16, color=THEME_COLORS['text']),
                                x=0.5,
                                xanchor='center'
                            ),
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig_dist)
                        
                        # Exibir estatísticas
                        st.subheader("Prediction Statistics")
                        stats_df = pd.DataFrame({
                            'Class': result_df['pc_class_desc'].value_counts().index,
                            'Count': result_df['pc_class_desc'].value_counts().values,
                            'Percentage': (result_df['pc_class_desc'].value_counts().values / len(result_df) * 100).round(2)
                        })
                        st.table(stats_df)
                        
                        # Exibir dados com previsões
                        st.subheader("Detailed Results")
                        st.dataframe(result_df)
                
                # Botão de download (só aparece se houver resultados)
                with col2:
                    if 'prediction_results' in st.session_state:
                        csv = st.session_state['prediction_results'].to_csv(index=False)
                        st.download_button(
                            label="Download Predictions (CSV)",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                            type="secondary"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer com o botão de download do dataset de exemplo
st.markdown("---")
footer_container = st.container()

with footer_container:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center'>
            <p>Developed with ❤️ using Streamlit and Plotly</p>
            <p>Data updated daily | Model: Random Forest Classifier</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Criar dataset de exemplo
        def create_sample_dataset():
            # Carregar dados originais
            original_data = load_data()
            
            # Selecionar algumas linhas aleatórias como exemplo
            sample_data = original_data.sample(n=10, random_state=42)
            
            # Remover a coluna de classe
            sample_data = sample_data.drop('pc_class', axis=1)
            
            return sample_data
        
        # Botão para download do dataset de exemplo
        sample_data = create_sample_dataset()
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Dataset",
            data=csv,
            file_name="sample_dataset.csv",
            mime="text/csv",
            help="Download a sample dataset with 10 companies to test the prediction tool",
            type="secondary"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)  # Espaço extra no final 