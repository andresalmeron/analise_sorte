import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # ESTA É A MÁGICA QUE IMPEDE O GRÁFICO DE SUMIR
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Sorte ou Habilidade?", layout="wide")

def processar_dados(df):
    """Limpa e formata o dataframe de cotas para retornos mensais."""
    # Pegar apenas as duas primeiras colunas (Data e Cota)
    df = df.iloc[:, :2]
    df.columns = ['Data', 'Cota']
    
    # Converter para datetime
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data'])
    
    # Garantir ordem cronológica (do mais antigo para o mais novo)
    df = df.sort_values('Data').reset_index(drop=True)
    
    # Converter cota para float (lidando com padrão brasileiro)
    if df['Cota'].dtype == object:
        df['Cota'] = df['Cota'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        
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
        # Leitura robusta para arquivos Comdinheiro/Economatica
        if arquivo_upload.name.endswith('.csv'):
            df_raw = pd.read_csv(arquivo_upload, sep=None, engine='python')
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
        
        with st.spinner('Gerando 10.000 universos paralelos (Bootstrapping)...'):
            # Embaralha os retornos criando N cenários
            indices_aleatorios = np.random.randint(0, n_meses, size=(n_meses, num_simulacoes))
            simulacoes_retornos = retornos_historicos[indices_aleatorios]
            
            # Calcula o retorno acumulado (com base 1.0 no mês 0)
            caminhos_acumulados = np.cumprod(simulacoes_retornos + 1, axis=0)
            caminhos_acumulados = np.vstack([np.ones(num_simulacoes), caminhos_acumulados]) # Adiciona origem comum
            
            # Trajetória real (com base 1.0 no mês 0)
            trajetoria_real = np.cumprod(retornos_historicos + 1)
            trajetoria_real = np.insert(trajetoria_real, 0, 1.0)
            
            # Criação da figura com o backend fixado
            fig, ax = plt.subplots(figsize=(12, 6))
            
            eixo_x = np.arange(n_meses + 1)
            
            # Plota uma amostra (ex: 500) para manter o gráfico legível e rápido
            ax.plot(eixo_x, caminhos_acumulados[:, :500], color='lightgray', alpha=0.15)
            
            # Plota a linha real por cima de tudo
            ax.plot(eixo_x, trajetoria_real, color='#004488', linewidth=3, label='Trajetória Real do Fundo')
            
            # Formatação visual
            ax.set_title(f'Trajetória Real vs. {num_simulacoes} Caminhos Aleatórios', fontsize=14)
            ax.set_ylabel('Fator de Crescimento do Capital')
            ax.set_xlabel('Meses Alocados')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        # Renderização explicitamente fora do spinner
        st.pyplot(fig)
        plt.close(fig) # Limpa a memória
        
        st.markdown("""
        **Como interpretar a imagem:**
        * A mancha cinza representa o **domínio da sorte**. Todos esses caminhos contêm exatamente as mesmas taxas de retorno que o fundo teve, apenas embaralhadas.
        * Se a linha azul não foge expressivamente da nuvem cinza, o resultado prático se deve fundamentalmente à exposição aos fatores de risco durante o período, e não à habilidade do gestor em fazer *market timing*.
        """)
            
        st.divider()

        # -------------------------------------------------------------------
        # SEÇÃO 3: A Régua da Dúvida (T-Stat)
        # -------------------------------------------------------------------
        st.subheader("3. A Régua da Dúvida: Estatística T")
        st.markdown("""
        A Estatística T traduz o gráfico acima em números: ela mede se o retorno gerado foi forte o suficiente para romper a barreira do ruído e ser considerado estatisticamente significativo.
        """)
        
        t_stat = retorno_medio_mensal / (volatilidade_mensal / np.sqrt(n_meses))
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
                st.info("O modelo assume que o Alfa é zero. A volatilidade exige tempo irrealista para confirmar consistência.")
        
    except Exception as e:
        st.error(f"Erro ao processar o arquivo. Detalhe técnico: {e}")
else:
    st.info("Aguardando o upload da base de dados (planilha) para iniciar a demonstração.")
