# Stock Market Analysis & Prediction App 📈

Uma aplicação web interativa construída com Streamlit para análise e previsão de desempenho de ações no mercado financeiro.

## 🌟 Funcionalidades

### 1. Market Analysis 📊
- Visualização interativa da relação entre Market Cap e P/E Ratio
- Tamanho dos pontos proporcional à receita da empresa
- Distribuição de classes de desempenho por país
- Todas as métricas em escala logarítmica para melhor visualização

### 2. Performance Metrics 📉
- Box plots das métricas financeiras por classe de desempenho
- Matriz de correlação interativa com valores numéricos
- Visualização em escala logarítmica para melhor comparação

### 3. Prediction Tool 🤖
- Interface para previsão individual de desempenho
- Entrada de métricas financeiras e macroeconômicas
- Visualização das probabilidades de cada classe

### 4. Batch Prediction 📋
- Upload de arquivo CSV para previsões em lote
- Visualização da distribuição das previsões
- Download dos resultados em CSV
- Estatísticas detalhadas das previsões

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Streamlit**: Interface web interativa
- **Plotly**: Visualizações interativas
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Modelo de Machine Learning (Random Forest)
- **Joblib**: Serialização do modelo

## 📦 Instalação

1. Clone o repositório:
```bash
git clone <repository-url>
cd streamlit_app
```

2. Crie um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🚀 Como Usar

1. Execute a aplicação:
```bash
streamlit run app.py
```

2. Acesse a aplicação no navegador (geralmente em http://localhost:8501)

### Para Previsões em Lote:

Prepare um arquivo CSV com as seguintes colunas:
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

Colunas opcionais:
- name
- country

## 🎨 Personalização

O tema da aplicação pode ser personalizado editando o arquivo `.streamlit/config.toml`.

## 📊 Modelo de Machine Learning

- **Algoritmo**: Random Forest Classifier
- **Classes de Previsão**:
  - 0: Baixo Desempenho
  - 1: Médio Desempenho
  - 2: Alto Desempenho

## 📝 Notas

- Os dados de entrada para previsões devem estar na mesma escala dos dados de treino
- Todas as visualizações usam escala logarítmica para melhor interpretação
- O modelo é treinado com pesos balanceados para evitar viés

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests.

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autores

- Seu Nome - Desenvolvimento inicial

## 🙏 Agradecimentos

- Streamlit pela excelente framework
- Plotly pela biblioteca de visualização
- Scikit-learn pela implementação do Random Forest 