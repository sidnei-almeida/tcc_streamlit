# Análise e Previsão de Crescimento Empresarial 📈

Uma aplicação web interativa construída com Streamlit para análise e previsão do potencial de crescimento de empresas, baseada em indicadores financeiros e macroeconômicos.

## 🌟 Funcionalidades

### 1. Visualização de Dados 📊
- Gráfico de dispersão interativo com indicadores financeiros
- Tamanho dos pontos proporcional à receita da empresa
- Cores indicam o potencial de crescimento
- Todas as métricas em escala logarítmica para melhor visualização
- Tooltips detalhados com informações da empresa

### 2. Previsão Individual 🎯
- Interface intuitiva para análise de uma empresa
- Entrada de métricas financeiras e macroeconômicas
- Visualização das probabilidades para cada classe de potencial
- Gráfico de barras com as probabilidades previstas

### 3. Previsão em Lote 📋
- Upload de arquivo CSV para análises em massa
- Template com 500 empresas de exemplo
- Análise estatística completa dos resultados:
  - Distribuição das classificações (gráfico de pizza)
  - Análise por país (gráfico de barras empilhadas)
  - Estatísticas das probabilidades
  - Correlações entre variáveis e probabilidades (mapa de calor)
- Download dos resultados completos em CSV

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
cd tcc_streamlit
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

1. Baixe o template com 500 empresas de exemplo na interface
2. Modifique o arquivo conforme necessário, mantendo as seguintes colunas:
```
- name (nome da empresa)
- country (país)
- dividend_yield_ttm (rendimento de dividendos)
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
- exchange_rate_to_usd (taxa de câmbio para USD)
```

## 📊 Modelo de Machine Learning

- **Algoritmo**: Random Forest Classifier
- **Classes de Previsão**:
  - Baixo Potencial de Crescimento
  - Médio Potencial de Crescimento
  - Alto Potencial de Crescimento

## 📝 Notas

- Os dados de entrada para previsões devem estar na mesma escala dos dados de treino
- Todas as visualizações usam escala logarítmica para melhor interpretação
- O modelo é treinado com pesos balanceados para evitar viés
- O dataset de exemplo é gerado sinteticamente com base nas distribuições dos dados reais

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests.

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autor

- Sidnei Almeida - Desenvolvimento 