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
    0: "Baixo Potencial de Crescimento",
    1: "Médio Potencial de Crescimento",
    2: "Alto Potencial de Crescimento"
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
    
    # Adicionar descrições das classes
    plot_data['pc_class_desc'] = plot_data['pc_class'].map(CLASS_MAPPING)
    
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
    page_title="Previsão de Potencial de Crescimento Empresarial",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Função para carregar os dados
@st.cache_data
def load_data():
    try:
        # Carregar dados do arquivo local
        data = pd.read_csv('data.csv')
        
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
        
    except FileNotFoundError:
        st.error("Arquivo data.csv não encontrado.")
        return None
    except pd.errors.EmptyDataError:
        st.error("O arquivo de dados está vazio.")
        return None
    except Exception as e:
        st.error(f"Erro ao processar os dados: {str(e)}")
        return None

# Criar dataset de exemplo
def create_sample_dataset():
    original_data = load_data()
    if original_data is not None:
        sample_data = original_data.sample(n=10, random_state=42)
        sample_data = sample_data.drop('pc_class', axis=1)
        return sample_data
    return None

# Estilo personalizado
st.markdown("""
    <style>
        /* Estilo para títulos */
        h1, h2, h3 {
            color: #0a4154;
            font-family: 'sans-serif';
        }
        
        /* Estilo para texto normal */
        p {
            color: #2d3748;
            font-family: 'sans-serif';
        }
        
        /* Estilo para widgets */
        .stSelectbox, .stSlider {
            color: #0a4154;
        }
        
        /* Reset de estilos para botões */
        .stButton>button,
        button[kind="secondary"] {
            all: unset;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #0a4154 !important;
            background-color: transparent;
            border: 2px solid #0a4154;
            border-radius: 4px;
            transition: all 0.3s ease;
            font-weight: 500;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            box-sizing: border-box;
            min-height: 40px;
        }
        
        /* Hover effect para botões */
        .stButton>button:hover,
        button[kind="secondary"]:hover {
            color: #ffffff !important;
            background-color: #0a4154;
            border-color: #0a4154;
        }
        
        /* Garantir que o texto dentro do botão também mude de cor */
        .stButton>button:hover *,
        button[kind="secondary"]:hover * {
            color: #ffffff !important;
        }
        
        /* Estilo para cards/containers */
        .stMarkdown {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        
        /* Estilo para métricas */
        .stMetric {
            color: #0a4154;
        }

        /* Estilo para a sidebar */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            padding: 2rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar com informações e download
with st.sidebar:
    st.markdown("""
        <div style='margin-bottom: 2rem;'>
            <h3 style='color: #0a4154; font-size: 1.2rem; margin-bottom: 1rem;'>
                Sobre a Ferramenta
            </h3>
            <p style='color: #2d3748; font-size: 0.9rem; margin-bottom: 1rem;'>
                Esta ferramenta utiliza Machine Learning para analisar e prever o potencial de crescimento de empresas com base em indicadores financeiros e econômicos.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background-color: #ffffff; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;'>
            <h4 style='color: #0a4154; font-size: 1rem; margin-bottom: 0.5rem;'>
                Dataset de Exemplo
            </h4>
            <p style='color: #2d3748; font-size: 0.85rem; margin-bottom: 1rem;'>
                Baixe um conjunto de dados de exemplo para testar a ferramenta de previsão.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Criar dataset de exemplo e adicionar botão de download logo após o texto
    sample_data = create_sample_dataset()
    if sample_data is not None:
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Baixar Exemplo",
            data=csv,
            file_name="dados_exemplo.csv",
            mime="text/csv",
            help="Dataset com 10 empresas para teste",
            type="secondary",
            key="download_example_sidebar"
        )

    st.markdown("---")
    st.markdown("""
        <div style='margin-top: 2rem;'>
            <p style='color: #718096; font-size: 0.8rem; text-align: center;'>
                Dados atualizados diariamente<br>
                Modelo: Random Forest Classifier
            </p>
        </div>
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

# Título principal
st.title("Análise e Previsão de Potencial de Crescimento Empresarial")
st.markdown("""
    <p style='font-size: 1.1rem; color: #2d3748; margin-bottom: 2rem;'>
        Utilize dados financeiros e econômicos para avaliar o potencial de crescimento de empresas através de análise preditiva.
    </p>
""", unsafe_allow_html=True)
st.markdown("---")

if data is not None:
    # Preparar dados para visualização
    plot_data = prepare_plot_data(data)
    
    # Métricas principais com três colunas
    col1, col2, col3 = st.columns(3)

    # Métricas principais
    with col1:
        st.metric(
            "Total de Empresas",
            len(data)
        )

    with col2:
        avg_market_cap = data['marketcap'].mean()
        st.metric(
            "Valor de Mercado Médio",
            format_large_number(avg_market_cap)
        )

    with col3:
        avg_pe = data['pe_ratio_ttm'].mean()
        st.metric(
            "Índice P/L Médio",
            format_large_number(avg_pe)
        )

    # Seção de download do dataset de exemplo
    st.markdown("---")
    
    # Criar abas
    tab1, tab2, tab3 = st.tabs([
        "📊 Visualização de Dados",
        "🎯 Previsão Individual",
        "📑 Previsão em Lote"
    ])

    with tab1:
        st.header("Visualização e Análise de Indicadores de Crescimento")
        
        # Seleção de variáveis para o gráfico
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox(
                "Selecione o indicador para o eixo X",
                options=['marketcap', 'revenue_ttm', 'earnings_ttm', 'price', 'pe_ratio_ttm', 'dividend_yield_ttm'],
                format_func=lambda x: {
                    'marketcap': 'Valor de Mercado',
                    'revenue_ttm': 'Receita',
                    'earnings_ttm': 'Lucros',
                    'price': 'Preço',
                    'pe_ratio_ttm': 'Índice P/L',
                    'dividend_yield_ttm': 'Rendimento de Dividendos'
                }[x]
            )
        
        with col2:
            y_var = st.selectbox(
                "Selecione a variável para o eixo Y",
                options=['marketcap', 'revenue_ttm', 'earnings_ttm', 'price', 'pe_ratio_ttm', 'dividend_yield_ttm'],
                index=1,
                format_func=lambda x: {
                    'marketcap': 'Valor de Mercado',
                    'revenue_ttm': 'Receita',
                    'earnings_ttm': 'Lucros',
                    'price': 'Preço',
                    'pe_ratio_ttm': 'Índice P/L',
                    'dividend_yield_ttm': 'Rendimento de Dividendos'
                }[x]
            )
        
        # Criar o gráfico de dispersão
        fig = px.scatter(
            plot_data,
            x=f"{x_var}_log",
            y=f"{y_var}_log",
            color='pc_class_desc',
            size='size_value',
            hover_data={
                f"{x_var}_log": False,
                f"{y_var}_log": False,
                x_var: ':,.2f',
                y_var: ':,.2f'
            },
            color_discrete_sequence=THEME_COLORS['marker_colors'],
            title=f'Relação entre Indicadores de Crescimento',
            labels={
                f"{x_var}_log": {
                    'marketcap': 'Valor de Mercado (log)',
                    'revenue_ttm': 'Receita (log)',
                    'earnings_ttm': 'Lucros (log)',
                    'price': 'Preço (log)',
                    'pe_ratio_ttm': 'Índice P/L (log)',
                    'dividend_yield_ttm': 'Rendimento de Dividendos (log)'
                }[x_var],
                f"{y_var}_log": {
                    'marketcap': 'Valor de Mercado (log)',
                    'revenue_ttm': 'Receita (log)',
                    'earnings_ttm': 'Lucros (log)',
                    'price': 'Preço (log)',
                    'pe_ratio_ttm': 'Índice P/L (log)',
                    'dividend_yield_ttm': 'Rendimento de Dividendos (log)'
                }[y_var],
                'pc_class_desc': 'Potencial de Crescimento'
            }
        )

        # Atualizar o layout do gráfico
        fig.update_layout(
            template=None,
            plot_bgcolor=THEME_COLORS['background'],
            paper_bgcolor=THEME_COLORS['background'],
            title=dict(
                font=dict(size=16, color=THEME_COLORS['text']),
                x=0.5,
                xanchor='center'
            ),
            legend=dict(
                title=dict(text='Potencial de Crescimento', font=dict(color=THEME_COLORS['text'], size=12)),
                font=dict(color=THEME_COLORS['text'], size=10),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=THEME_COLORS['text'],
                borderwidth=1
            ),
            xaxis=dict(
                title_font=dict(size=12, color=THEME_COLORS['text']),
                tickfont=dict(size=10, color=THEME_COLORS['text']),
                gridcolor=THEME_COLORS['grid'],
                showgrid=True
            ),
            yaxis=dict(
                title_font=dict(size=12, color=THEME_COLORS['text']),
                tickfont=dict(size=10, color=THEME_COLORS['text']),
                gridcolor=THEME_COLORS['grid'],
                showgrid=True
            ),
            margin=dict(t=50, b=20, l=20, r=20),  # Aumentar margem superior
            height=500  # Definir altura fixa um pouco maior para o gráfico de dispersão
        )

        # Exibir o gráfico
        st.plotly_chart(fig, use_container_width=True)

        # Adicionar explicação sobre o gráfico
        st.markdown("""
        **Sobre o gráfico:**
        - Cada ponto representa uma empresa
        - O tamanho do ponto é proporcional à receita da empresa
        - As cores indicam o potencial de crescimento previsto
        - Os valores estão em escala logarítmica para melhor visualização
        """)

    with tab2:
        st.header("Previsão Individual")
        st.markdown("""
            <p style='color: #2d3748; font-size: 0.95rem; margin-bottom: 1.5rem;'>
                Insira os dados de uma empresa para obter uma previsão de seu potencial de crescimento.
            </p>
        """, unsafe_allow_html=True)

        # Formulário de entrada de dados
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Nome da Empresa")
                country = st.text_input("País")
                marketcap = st.number_input("Valor de Mercado", min_value=0.0, format="%.2f")
                pe_ratio_ttm = st.number_input("Índice P/L", min_value=0.0, format="%.2f")
                revenue_ttm = st.number_input("Receita", min_value=0.0, format="%.2f")
                price = st.number_input("Preço", min_value=0.0, format="%.2f")
            
            with col2:
                earnings_ttm = st.number_input("Lucros", min_value=0.0, format="%.2f")
                dividend_yield_ttm = st.number_input("Rendimento de Dividendos", min_value=0.0, format="%.2f")
                gdp_per_capita_usd = st.number_input("PIB per Capita", min_value=0.0, format="%.2f")
                gdp_growth_percent = st.number_input("Taxa de Crescimento do PIB", min_value=0.0, format="%.2f")
                inflation_percent = st.number_input("Taxa de Inflação", min_value=0.0, format="%.2f")
                interest_rate_percent = st.number_input("Taxa de Juros", min_value=0.0, format="%.2f")
                unemployment_rate_percent = st.number_input("Taxa de Desemprego", min_value=0.0, format="%.2f")
                exchange_rate_to_usd = st.number_input("Câmbio para USD", min_value=0.0, format="%.2f")
            
            submit_button = st.form_submit_button("Obter Previsão")

        if submit_button:
            # Preparar dados de entrada
            input_data = pd.DataFrame({
                'name': [name],
                'country': [country],
                'marketcap': [marketcap],
                'pe_ratio_ttm': [pe_ratio_ttm],
                'revenue_ttm': [revenue_ttm],
                'price': [price],
                'earnings_ttm': [earnings_ttm],
                'dividend_yield_ttm': [dividend_yield_ttm],
                'gdp_per_capita_usd': [gdp_per_capita_usd],
                'gdp_growth_percent': [gdp_growth_percent],
                'inflation_percent': [inflation_percent],
                'interest_rate_percent': [interest_rate_percent],
                'unemployment_rate_percent': [unemployment_rate_percent],
                'exchange_rate_to_usd': [exchange_rate_to_usd]
            })
            
            # Fazer previsão
            prediction = batch_predict(input_data)
            
            if prediction is not None:
                # Exibir resultado
                st.subheader("Resultado da Previsão")
                st.write(f"Potencial de Crescimento: {prediction['pc_class_desc'].iloc[0]}")
                
                # Exibir probabilidades
                st.subheader("Probabilidades")
                prob_data = prediction[['prob_Baixo Potencial de Crescimento', 'prob_Médio Potencial de Crescimento', 'prob_Alto Potencial de Crescimento']].iloc[0]
                prob_fig = px.bar(
                    prob_data,
                    x=prob_data.index,
                    y=prob_data.values,
                    color=prob_data.index,
                    color_discrete_sequence=THEME_COLORS['marker_colors'],
                    labels={
                        'x': 'Potencial de Crescimento',
                        'y': 'Probabilidade'
                    }
                )
                
                prob_fig.update_layout(
                    showlegend=False,
                    title_x=0.5,
                    title_font=dict(size=14, color=THEME_COLORS['text']),
                    xaxis_title="",
                    yaxis_title="Probabilidade",
                    plot_bgcolor=THEME_COLORS['background'],
                    paper_bgcolor=THEME_COLORS['background'],
                    margin=dict(t=50, b=20, l=20, r=20),  # Aumentar margem superior
                    height=400  # Definir altura fixa
                )
                
                st.plotly_chart(prob_fig, use_container_width=True)

    with tab3:
        st.header("Previsão em Lote")
        st.markdown("""
            <p style='color: #2d3748; font-size: 0.95rem; margin-bottom: 1.5rem;'>
                Faça upload de um arquivo CSV contendo múltiplas empresas para análise em lote.
                O arquivo deve conter as mesmas colunas do dataset de exemplo.
            </p>
        """, unsafe_allow_html=True)

        # Botão para baixar exemplo na aba de previsão em lote
        if sample_data is not None:
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Baixar Template",
                data=csv,
                file_name="template_dados.csv",
                mime="text/csv",
                help="Template CSV com as colunas necessárias",
                type="secondary",
                key="download_example_batch"
            )

        # Upload do arquivo
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

        if uploaded_file is not None:
            try:
                # Carregar dados do arquivo
                input_data = pd.read_csv(uploaded_file)
                
                # Verificar colunas necessárias
                required_columns = [
                    'name', 'country',
                    'dividend_yield_ttm', 'earnings_ttm', 'marketcap',
                    'pe_ratio_ttm', 'revenue_ttm', 'price',
                    'gdp_per_capita_usd', 'gdp_growth_percent', 'inflation_percent',
                    'interest_rate_percent', 'unemployment_rate_percent', 'exchange_rate_to_usd'
                ]
                
                missing_columns = set(required_columns) - set(input_data.columns)
                if missing_columns:
                    st.error(f"Colunas ausentes no arquivo: {', '.join(missing_columns)}")
                else:
                    # Fazer previsões
                    with st.spinner("Processando dados..."):
                        predictions = batch_predict(input_data)
                    
                    if predictions is not None:
                        # Exibir resultados
                        st.subheader("Resultados da Previsão")
                        
                        # Criar um DataFrame simplificado para exibição
                        display_df = predictions[[
                            'name', 'country', 'pc_class_desc',
                            'prob_Baixo Potencial de Crescimento',
                            'prob_Médio Potencial de Crescimento',
                            'prob_Alto Potencial de Crescimento'
                        ]].copy()
                        
                        # Renomear colunas para melhor visualização
                        display_df.columns = [
                            'Empresa', 'País', 'Potencial de Crescimento',
                            'Prob. Baixo', 'Prob. Médio', 'Prob. Alto'
                        ]
                        
                        # Formatar colunas de probabilidade como percentual
                        for col in ['Prob. Baixo', 'Prob. Médio', 'Prob. Alto']:
                            display_df[col] = display_df[col].map('{:.1%}'.format)
                        
                        # Exibir tabela com resultados
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Análise Estatística dos Resultados
                        st.subheader("Análise Estatística dos Resultados")
                        
                        # Distribuição das classes
                        st.write("##### Distribuição das Classificações")
                        class_dist = predictions['pc_class_desc'].value_counts()
                        class_dist_df = pd.DataFrame({
                            'Classificação': class_dist.index,
                            'Quantidade': class_dist.values,
                            'Percentual': (class_dist.values / len(predictions) * 100).round(1)
                        })
                        
                        # Criar gráfico de pizza para distribuição das classes
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=class_dist_df['Classificação'],
                            values=class_dist_df['Quantidade'],
                            hole=0.4,
                            textinfo='label+percent',
                            textposition='outside',
                            pull=[0.1 if x == class_dist_df['Quantidade'].max() else 0 for x in class_dist_df['Quantidade']]
                        )])
                        fig_pie.update_layout(
                            title="Distribuição das Classificações",
                            height=400,
                            margin=dict(t=50, b=0, l=0, r=0),
                            showlegend=False
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Análise por País
                        if len(predictions['country'].unique()) > 1:
                            st.write("##### Análise por País")
                            country_analysis = pd.crosstab(
                                predictions['country'],
                                predictions['pc_class_desc'],
                                normalize='index'
                            ) * 100
                            
                            # Gráfico de barras empilhadas por país
                            fig_country = go.Figure()
                            for col in country_analysis.columns:
                                fig_country.add_trace(go.Bar(
                                    name=col,
                                    x=country_analysis.index,
                                    y=country_analysis[col],
                                    text=country_analysis[col].round(1).astype(str) + '%',
                                    textposition='auto',
                                ))
                            
                            fig_country.update_layout(
                                title="Distribuição das Classificações por País",
                                yaxis_title="Percentual",
                                barmode='stack',
                                height=400,
                                margin=dict(t=50, b=0, l=0, r=0)
                            )
                            st.plotly_chart(fig_country, use_container_width=True)
                        
                        # Estatísticas das probabilidades
                        st.write("##### Estatísticas das Probabilidades")
                        prob_cols = [
                            'prob_Baixo Potencial de Crescimento',
                            'prob_Médio Potencial de Crescimento',
                            'prob_Alto Potencial de Crescimento'
                        ]
                        prob_stats = predictions[prob_cols].agg(['mean', 'std', 'min', 'max']).round(3)
                        prob_stats.columns = ['Baixo Potencial', 'Médio Potencial', 'Alto Potencial']
                        prob_stats.index = ['Média', 'Desvio Padrão', 'Mínimo', 'Máximo']
                        
                        # Formatar as estatísticas como percentual
                        prob_stats_formatted = prob_stats.applymap(lambda x: f"{x*100:.1f}%")
                        st.dataframe(
                            prob_stats_formatted,
                            use_container_width=True
                        )
                        
                        # Correlações entre variáveis numéricas e probabilidades
                        st.write("##### Correlações com Probabilidades")
                        numeric_cols = [
                            'dividend_yield_ttm', 'earnings_ttm', 'marketcap',
                            'pe_ratio_ttm', 'revenue_ttm', 'price',
                            'gdp_per_capita_usd', 'gdp_growth_percent', 'inflation_percent',
                            'interest_rate_percent', 'unemployment_rate_percent', 'exchange_rate_to_usd'
                        ]
                        
                        correlations = predictions[numeric_cols + prob_cols].corr().loc[numeric_cols, prob_cols]
                        correlations.columns = ['Prob. Baixo', 'Prob. Médio', 'Prob. Alto']
                        
                        # Criar mapa de calor das correlações
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=correlations.values,
                            x=correlations.columns,
                            y=correlations.index,
                            text=correlations.values.round(2),
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            hoverongaps=False,
                            colorscale='RdBu',
                            zmin=-1,
                            zmax=1
                        ))
                        
                        fig_corr.update_layout(
                            title="Correlações entre Variáveis e Probabilidades",
                            height=600,
                            margin=dict(t=50, b=0, l=0, r=0)
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Adicionar botão para download dos resultados completos
                        csv_results = predictions.to_csv(index=False)
                        st.download_button(
                            label="📥 Baixar Resultados Completos",
                            data=csv_results,
                            file_name="resultados_previsao.csv",
                            mime="text/csv",
                            help="Baixar resultados completos em formato CSV",
                            type="secondary",
                            key="download_results"
                        )
                        
            except Exception as e:
                st.error(f"Erro ao processar o arquivo: {str(e)}")

# Footer
st.markdown("---")
footer_container = st.container()

with footer_container:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <p style='color: #2d3748; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                    Desenvolvido com ❤️ usando Streamlit e Plotly
                </p>
                <p style='color: #718096; font-size: 0.8rem;'>
                    Dados atualizados diariamente | Modelo: Random Forest Classifier
                </p>
                <p style='color: #718096; font-size: 0.8rem;'>
                    © 2024 Análise e Previsão do Mercado de Ações
                </p>
            </div>
        """, unsafe_allow_html=True) 