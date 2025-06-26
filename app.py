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
def generate_company_name():
    # Indústrias/Setores
    industries = [
        "Tech", "Global", "Smart", "Digital", "Advanced", "Future", "Innovative", "Modern",
        "Precision", "Dynamic", "Strategic", "Quantum", "Cyber", "Bio", "Green", "Eco",
        "Solar", "Cloud", "Data", "AI", "Nano", "Meta", "Nexus", "Core", "Prime"
    ]
    
    # Palavras de negócios
    business_words = [
        "Solutions", "Systems", "Technologies", "Industries", "Enterprises", "Networks",
        "Dynamics", "Analytics", "Innovations", "Applications", "Platforms", "Services",
        "Communications", "Engineering", "Software", "Hardware", "Robotics", "Energy",
        "Materials", "Healthcare", "Ventures", "Capital", "Group", "Corporation"
    ]
    
    # Sufixos comuns
    suffixes = [
        "Corp", "Inc", "Ltd", "Group", "Holdings", "International", "Global",
        "Technologies", "Solutions", "Enterprises", "Systems", "Partners"
    ]
    
    # Gerar nome
    name_parts = []
    
    # 50% de chance de usar duas palavras da indústria
    if np.random.random() < 0.5:
        name_parts.extend(np.random.choice(industries, size=2, replace=False))
    else:
        name_parts.append(np.random.choice(industries))
    
    # Adicionar palavra de negócios
    name_parts.append(np.random.choice(business_words))
    
    # 30% de chance de adicionar sufixo
    if np.random.random() < 0.3:
        name_parts.append(np.random.choice(suffixes))
    
    return " ".join(name_parts)

def create_sample_dataset():
    original_data = load_data()
    if original_data is not None:
        # Criar um conjunto maior de dados sintéticos baseado nos dados originais
        sample_data = pd.DataFrame()
        
        # Número de empresas desejado
        n_samples = 500
        
        # Para cada coluna, gerar dados sintéticos baseados na distribuição dos dados originais
        for col in original_data.columns:
            if col == 'pc_class':
                continue
            elif col == 'name':
                # Gerar nomes únicos de empresas
                sample_data[col] = [generate_company_name() for _ in range(n_samples)]
            elif col == 'country':
                # Para país, apenas amostrar dos existentes
                values = original_data[col].values
                sample_data[col] = np.random.choice(values, size=n_samples)
            else:
                # Para variáveis numéricas, usar distribuição normal baseada nos dados originais
                mean = original_data[col].mean()
                std = original_data[col].std()
                # Gerar valores positivos para métricas que não podem ser negativas
                if col in ['marketcap', 'price', 'revenue_ttm', 'gdp_per_capita_usd']:
                    sample_data[col] = np.abs(np.random.normal(mean, std, n_samples))
                else:
                    sample_data[col] = np.random.normal(mean, std, n_samples)
        
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
        
        # Verificar se as colunas necessárias existem
        train_data = load_data()
        train_features = train_data.drop(['name', 'country', 'pc_class'], axis=1).columns
        
        # Verificar colunas ausentes
        missing_cols = set(train_features) - set(input_data.columns)
        if missing_cols:
            # Tentar mapear colunas antigas para novas
            column_mapping = {
                'inflation': 'inflation_percent',
                'interest_rate': 'interest_rate_percent',
                'unemployment': 'unemployment_rate_percent'
            }
            
            # Renomear colunas se existirem
            for old_col, new_col in column_mapping.items():
                if old_col in input_data.columns and new_col in missing_cols:
                    input_data[new_col] = input_data[old_col]
                    input_data.drop(old_col, axis=1, inplace=True)
            
            # Verificar novamente colunas ausentes
            missing_cols = set(train_features) - set(input_data.columns)
            if missing_cols:
                raise ValueError(f"Colunas ausentes: {', '.join(missing_cols)}")
        
        # Preparar features para previsão
        if 'name' in input_data.columns and 'country' in input_data.columns:
            features_df = input_data.drop(['name', 'country'], axis=1, errors='ignore')
        else:
            features_df = input_data
            
        # Reordenar colunas para match com treino
        features_df = features_df[train_features]
        
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
        "Visualização de Dados",
        "Previsão Individual",
        "Previsão em Lote"
    ])

    # Conteúdo da aba de Visualização de Dados
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
                y=0.95
            ),
            legend=dict(
                title=None,
                orientation="h",
                y=-0.15,
                yanchor="top",
                x=0.5,
                xanchor="center"
            ),
            margin=dict(t=50, b=100),
            height=600
        )
        
        # Exibir o gráfico
        st.plotly_chart(fig, use_container_width=True)
        
        # Adicionar explicação sobre a escala logarítmica
        st.markdown("""
        **Nota sobre a visualização:**
        - O tamanho dos pontos é proporcional à receita da empresa
        - Os valores estão em escala logarítmica para melhor visualização
        """)
        
    # Conteúdo da aba de Previsão Individual
    with tab2:
        st.header("Previsão Individual")
        st.markdown("""
            Insira os dados da empresa para obter uma previsão do seu potencial de crescimento.
            Todos os campos são obrigatórios.
        """)
        
        # Criar formulário para entrada de dados
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Nome da Empresa")
                country = st.text_input("País")
                dividend_yield_ttm = st.number_input("Dividend Yield (TTM)", min_value=0.0, format="%.6f")
                earnings_ttm = st.number_input("Lucros (TTM)", format="%.2f")
                marketcap = st.number_input("Valor de Mercado", min_value=0.0, format="%.2f")
                pe_ratio_ttm = st.number_input("Índice P/L (TTM)", format="%.2f")
                revenue_ttm = st.number_input("Receita (TTM)", format="%.2f")
            
            with col2:
                price = st.number_input("Preço", min_value=0.0, format="%.2f")
                gdp_per_capita_usd = st.number_input("PIB per Capita (USD)", min_value=0.0, format="%.2f")
                gdp_growth_percent = st.number_input("Crescimento do PIB (%)", format="%.2f")
                inflation_percent = st.number_input("Taxa de Inflação", min_value=0.0, format="%.2f")
                interest_rate_percent = st.number_input("Taxa de Juros", min_value=0.0, format="%.2f")
                unemployment_rate_percent = st.number_input("Taxa de Desemprego", min_value=0.0, format="%.2f")
                exchange_rate_to_usd = st.number_input("Taxa de Câmbio (USD)", min_value=0.0, format="%.2f")
            
            # Botão de submit
            submitted = st.form_submit_button("Fazer Previsão")
        
        if submitted:
            # Criar DataFrame com os dados de entrada
            input_data = pd.DataFrame({
                'name': [name],
                'country': [country],
                'dividend_yield_ttm': [dividend_yield_ttm],
                'earnings_ttm': [earnings_ttm],
                'marketcap': [marketcap],
                'pe_ratio_ttm': [pe_ratio_ttm],
                'revenue_ttm': [revenue_ttm],
                'price': [price],
                'gdp_per_capita_usd': [gdp_per_capita_usd],
                'gdp_growth_percent': [gdp_growth_percent],
                'inflation_percent': [inflation_percent],
                'interest_rate_percent': [interest_rate_percent],
                'unemployment_rate_percent': [unemployment_rate_percent],
                'exchange_rate_to_usd': [exchange_rate_to_usd]
            })
            
            # Fazer previsão
            with st.spinner("Processando..."):
                predictions = batch_predict(input_data)
            
            if predictions is not None:
                # Exibir resultados
                st.subheader("Resultado da Previsão")
                
                # Criar colunas para métricas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Classificação",
                        predictions['pc_class_desc'].iloc[0]
                    )
                
                with col2:
                    highest_prob = max([
                        predictions['prob_Baixo Potencial de Crescimento'].iloc[0],
                        predictions['prob_Médio Potencial de Crescimento'].iloc[0],
                        predictions['prob_Alto Potencial de Crescimento'].iloc[0]
                    ])
                    st.metric(
                        "Confiança",
                        f"{highest_prob*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Classe",
                        f"Classe {predictions['pc_class'].iloc[0]}"
                    )
                
                # Criar gráfico de probabilidades
                prob_data = {
                    'Classe': ['Baixo', 'Médio', 'Alto'],
                    'Probabilidade': [
                        predictions['prob_Baixo Potencial de Crescimento'].iloc[0],
                        predictions['prob_Médio Potencial de Crescimento'].iloc[0],
                        predictions['prob_Alto Potencial de Crescimento'].iloc[0]
                    ]
                }
                
                prob_fig = go.Figure(data=[
                    go.Bar(
                        x=prob_data['Classe'],
                        y=prob_data['Probabilidade'],
                        text=[f"{p*100:.1f}%" for p in prob_data['Probabilidade']],
                        textposition='auto',
                        marker_color=THEME_COLORS['marker_colors']
                    )
                ])
                
                prob_fig.update_layout(
                    title="Probabilidades por Classe",
                    xaxis_title="Potencial de Crescimento",
                    yaxis_title="Probabilidade",
                    yaxis=dict(
                        tickformat=".0%",
                        range=[0, 1]
                    ),
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(prob_fig, use_container_width=True)
                
    # Conteúdo da aba de Previsão em Lote
    with tab3:
        st.header("Previsão em Lote")
        st.markdown("""
            Faça upload de um arquivo CSV contendo múltiplas empresas para análise.
            O arquivo deve conter as seguintes colunas:
            - name (nome da empresa)
            - country (país)
            - dividend_yield_ttm (dividend yield)
            - earnings_ttm (lucros)
            - marketcap (valor de mercado)
            - pe_ratio_ttm (índice P/L)
            - revenue_ttm (receita)
            - price (preço)
            - gdp_per_capita_usd (PIB per capita)
            - gdp_growth_percent (crescimento do PIB)
            - inflation_percent (inflação)
            - interest_rate_percent (taxa de juros)
            - unemployment_rate_percent (taxa de desemprego)
            - exchange_rate_to_usd (taxa de câmbio)
        """)
        
        # Criar duas colunas
        col1, col2 = st.columns(2)
        
        with col1:
            # Botão para baixar template
            sample_data = create_sample_dataset()
            if sample_data is not None:
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="📥 Baixar Template com Exemplos",
                    data=csv,
                    file_name="template_empresas.csv",
                    mime="text/csv",
                    help="Baixar arquivo CSV de exemplo com 500 empresas",
                    type="secondary",
                    key="download_template"
                )

        # Upload do arquivo
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv", key="batch_file_uploader")
        
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
                    # Mostrar preview dos dados
                    st.write("Preview dos dados carregados:")
                    st.dataframe(
                        input_data.head(),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Botão para fazer a previsão
                    if st.button("Realizar Previsão", type="primary", key="batch_predict_button"):
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
                                    title="Distribuição por País",
                                    barmode='stack',
                                    height=400,
                                    yaxis_title="Percentual",
                                    xaxis_title="País",
                                    showlegend=True
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
                    Modelo: Random Forest Classifier
                </p>
                <p style='color: #718096; font-size: 0.8rem;'>
                    © 2024 Análise e Previsão de Crescimento Empresarial
                </p>
            </div>
        """, unsafe_allow_html=True) 