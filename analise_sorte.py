import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Sorte ou Habilidade?", layout="wide")

def processar_dados(df):
    """Limpa e formata o dataframe de cotas para retornos mensais."""
    # Renomear colunas para padrão assumindo que a 1ª é Data e a 2ª é Cota
    df.columns = ['Data', 'Cota']
    
    # Converter para datetime
    df['Data'] = pd.to_datetime(df['Data'])
    
    # Garantir ordem cronológica (do mais antigo para o mais novo)
    df = df.sort_values('Data').reset_index(drop=True)
    
    # Converter cota para float (removendo possíveis strings ou vírgulas se houver)
    if df['Cota'].dtype == object:
        df['Cota'] = df['Cota'].astype(str).str.replace(',', '.').astype(float)
        
    # Calcular o retorno percentual mensal
    df['Retorno'] = df['Cota'].pct_change()
    
    # Remover o primeiro valor que será NaN
    df = df.dropna().reset_index(drop=True)
    
    return df

st.title("Sorte ou Habilidade? O Teste da Aleatoriedade")
st.markdown("""
Esta ferramenta aplica testes estatísticos rigorosos para responder a uma pergunta fundamental na alocação de capital: **O retorno histórico de um fundo é fruto da genialidade do gestor ou apenas um passeio aleatório (Random Walk) guiado pelo ruído do mercado?**
""")

# Upload do arquivo
arquivo_upload = st.file_uploader("Faça o upload da planilha de cotas (CSV ou Excel)", type=['csv', 'xlsx'])

if arquivo_upload is not None:
    try:
        # Tenta ler como CSV primeiro, se falhar, tenta Excel
        if arquivo_upload.name.endswith('.csv'):
            df_raw = pd.read_csv(arquivo_upload)
        else:
            df_raw = pd.read_excel(arquivo_upload)
            
        df = processar_dados(df_raw)
        
        # Estatísticas descritivas básicas
        n_meses = len(df)
        anos = n_meses / 12
        retorno_medio_mensal = df['Retorno'].mean()
        volatilidade_mensal = df['Retorno'].std()
        
        retorno_anualizado = ((1 + retorno_medio_mensal) ** 12) - 1
        volatilidade_anualizada = volatilidade_mensal * np.sqrt(12)
        
        st.subheader("1. Resumo da Amostra")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Período Analisado", f"{anos:.1f} anos")
        col2.metric("Meses (N)", f"{n_meses}")
        col3.metric("Retorno Médio (a.a.)", f"{retorno_anualizado:.2%}")
        col4.metric("Volatilidade (a.a.)", f"{volatilidade_anualizada:.2%}")
        
        st.divider()

        # -------------------------------------------------------------------
        # SEÇÃO 2: O Multiverso do Azar (Monte Carlo)
        # -------------------------------------------------------------------
        st.subheader("2. O Multiverso do Azar: Simulação de Monte Carlo")
        st.markdown("""
        Se o mercado é um passeio aleatório, o que acontece se pegarmos todos os retornos mensais deste fundo e **sortearmos a ordem deles 10.000 vezes**? 
        
        Se a linha azul (o histórico real do fundo) terminar no meio da nuvem cinza (os caminhos aleatórios), significa que **milhares de investidores jogando dados chegariam ao mesmo resultado**. A consistência da trajetória, neste caso, é uma ilusão.
        """)
        
        num_simulacoes = 10000
        retornos_historicos = df['Retorno'].values
        
        # O processamento matemático fica dentro do spinner
        with st.spinner('Gerando 10.000 universos paralelos (Bootstrapping)...'):
            # Gera matriz de índices aleatórios com reposição
            indices_aleatorios = np.random.randint(0, n_meses, size=(n_meses, num_simulacoes))
            
            # Mapeia os índices para os retornos reais
            simulacoes_retornos = retornos_historicos[indices_aleatorios]
            
            # Adiciona 1 para calcular retorno acumulado
            simulacoes_retornos_mais_um = simulacoes_retornos + 1
            
            # Calcula o retorno acumulado de cada caminho
            caminhos_acumulados = np.cumprod(simulacoes_retornos_mais_um, axis=0)
            
            # Trajetória real para comparação
            trajetoria_real = np.cumprod(retornos_historicos + 1)
            
            # Criação da figura
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plota uma amostra dos caminhos (ex: 500) para manter o navegador fluido
            ax.plot(caminhos_acumulados[:, :500], color='lightgray', alpha=0.15)
            
            # Plota a linha real
            ax.plot(trajetoria_real, color='#004488', linewidth=3, label='Trajetória Real do Fundo')
            
            # Formatação do gráfico
            ax.set_title(f'Trajetória Real vs. Caminhos Aleatórios ({n_meses} meses)', fontsize=14)
            ax.set_ylabel('Fator de Crescimento do Capital')
            ax.set_xlabel('Meses')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend()
            
            # Design mais limpo
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        # A renderização do gráfico fica FORA do spinner para garantir que não desapareça
        st.pyplot(fig)
        
        # Conclusão pedagógica
        st.markdown("""
        **Como interpretar a imagem:**
        * A mancha cinza representa o **domínio da sorte**. Todos esses caminhos contêm exatamente as mesmas taxas de retorno que o fundo teve, apenas em ordens diferentes.
        * Se a linha azul não foge expressivamente da nuvem cinza, o resultado prático se deve fundamentalmente à exposição sistemática aos fatores de risco, e não à habilidade do gestor de "bater o mercado".
        """)
            
        st.divider()

        # -------------------------------------------------------------------
        # SEÇÃO 3: A Régua da Dúvida (T-Stat)
        # -------------------------------------------------------------------
        st.subheader("3. A Régua da Dúvida: Estatística T")
        st.markdown("""
        Na análise quantitativa, assume-se que **o Alfa é zero até que se prove o contrário**. A Estatística T traduz o gráfico acima em números: ela mede se o retorno gerado foi forte o suficiente para romper a barreira do ruído de mercado e ser considerado estatisticamente significativo.
        """)
        
        # Cálculo do T-Stat considerando H0: Retorno Médio = 0
        t_stat = retorno_medio_mensal / (volatilidade_mensal / np.sqrt(n_meses))
        
        # Cálculo de quantos anos seriam necessários para t = 2.0 (95% de confiança)
        anos_necessarios = ((2.0 * volatilidade_mensal) / retorno_medio_mensal)**2 / 12 if retorno_medio_mensal > 0 else float('inf')
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("T-Stat (Ouro > 2.0)", f"{t_stat:.2f}")
            if t_stat >= 2.0:
                st.success("Resultado Estatisticamente Significativo. Há fortes indícios matemáticos contra o acaso puro.")
            else:
                st.warning("Resultado NÃO Significativo. O retorno médio não superou o ruído da amostra.")
                
        with c2:
            if retorno_medio_mensal > 0:
                st.metric("Anos de histórico exigidos para provar habilidade (T=2.0)", f"{anos_necessarios:.1f} anos")
                st.info("Isto mostra o tempo irrealista que a volatilidade exige para confirmar consistência.")
        
    except Exception as e:
        st.error(f"Erro ao processar o arquivo. Detalhe técnico: {e}")
else:
    st.info("Aguardando o upload da base de dados (planilha) para iniciar a demonstração.")
