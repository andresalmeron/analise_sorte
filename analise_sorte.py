import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Sorte ou Habilidade?", layout="wide")

def ler_arquivo(arquivo):
    if arquivo.name.endswith('.csv'):
        try:
            df = pd.read_csv(arquivo, sep=',', encoding='utf-8')
            if df.shape[1] < 2: 
                arquivo.seek(0)
                df = pd.read_csv(arquivo, sep=';', encoding='latin-1')
        except Exception:
            arquivo.seek(0)
            df = pd.read_csv(arquivo, sep=';', encoding='latin-1')
    else:
        df = pd.read_excel(arquivo)
        
    if df.shape[1] < 2:
        raise ValueError("O arquivo não possui duas colunas úteis (Data e Cota/Preço).")
    return df

def limpar_numero(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    val_str = str(val).strip()
    val_limpo = re.sub(r'[^\d.,-]', '', val_str)
    if val_limpo == '': return np.nan
    
    if ',' in val_limpo and '.' in val_limpo:
        if val_limpo.rfind(',') > val_limpo.rfind('.'):
            val_limpo = val_limpo.replace('.', '').replace(',', '.')
        else:
            val_limpo = val_limpo.replace(',', '')
    elif ',' in val_limpo:
        val_limpo = val_limpo.replace(',', '.')
        
    try:
        return float(val_limpo)
    except:
        return np.nan

def formatar_percentual_br(valor):
    return f"{valor * 100:.2f}%".replace('.', ',')

def formatar_decimal_br(valor):
    return f"{valor:.2f}".replace('.', ',')

st.title("Sorte ou Habilidade? O Teste da Aleatoriedade")
st.markdown("""
Esta ferramenta aplica testes rigorosos para responder a uma pergunta fundamental na alocação de capital: **O retorno histórico de um fundo é fruto de habilidade extraordinária do gestor ou apenas um passeio aleatório ditado pelo ruído do mercado?**
""")

arquivo_upload = st.file_uploader("Faça o upload da planilha (Excel ou CSV da extração com os PREÇOS)", type=['csv', 'xlsx'])

if arquivo_upload is not None:
    df_raw = ler_arquivo(arquivo_upload)

    with st.expander("🛠️ Modo Depuração (Leitura Bruta)"):
        st.dataframe(df_raw.head())

    try:
        df_completo = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed')].copy()
        df_completo = df_completo.iloc[:, :2]
        df_completo.columns = ['Data', 'Cota']
        
        df_completo['Data'] = pd.to_datetime(df_completo['Data'], dayfirst=True, errors='coerce')
        df_completo['Cota'] = df_completo['Cota'].apply(limpar_numero)
        
        df_completo = df_completo.dropna(subset=['Data', 'Cota'])
        df_completo = df_completo.sort_values('Data').reset_index(drop=True)
        
        df_completo['MesAno_Str'] = df_completo['Data'].dt.strftime('%m/%Y')
        opcoes_datas = df_completo['MesAno_Str'].drop_duplicates().tolist()
        
        st.divider()
        st.subheader("Filtro de Período")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            idx_fim_default = len(opcoes_datas) - 1
            data_fim_str = st.selectbox("1. Escolha o Fim:", options=opcoes_datas, index=idx_fim_default)
            
        idx_fim = opcoes_datas.index(data_fim_str)
        
        with col2:
            opcoes_atalho = ["Personalizado", "Último 1 ano", "Últimos 3 anos", "Últimos 5 anos", "Últimos 10 anos", "Desde o Início"]
            atalho = st.selectbox("2. Janela de Tempo:", options=opcoes_atalho, index=4) 
            
        if atalho == "Último 1 ano": idx_ini_calc = max(0, idx_fim - 12)
        elif atalho == "Últimos 3 anos": idx_ini_calc = max(0, idx_fim - 36)
        elif atalho == "Últimos 5 anos": idx_ini_calc = max(0, idx_fim - 60)
        elif atalho == "Últimos 10 anos": idx_ini_calc = max(0, idx_fim - 120)
        elif atalho == "Desde o Início": idx_ini_calc = 0
        else: idx_ini_calc = max(0, idx_fim - 120)

        with col3:
            if atalho == "Personalizado":
                data_inicio_str = st.selectbox("3. Escolha o Início:", options=opcoes_datas, index=idx_ini_calc)
            else:
                data_inicio_str = st.selectbox("3. Início (Automático):", options=[opcoes_datas[idx_ini_calc]], disabled=True)
                
        idx_ini_real = opcoes_datas.index(data_inicio_str)
        if idx_ini_real >= idx_fim:
            st.error("⚠️ O Mês/Ano Inicial deve ser obrigatoriamente anterior ao Mês/Ano Final.")
            st.stop()
        
        data_inicio = df_completo[df_completo['MesAno_Str'] == data_inicio_str]['Data'].min()
        data_fim = df_completo[df_completo['MesAno_Str'] == data_fim_str]['Data'].max()
        
        mask = (df_completo['Data'] >= data_inicio) & (df_completo['Data'] <= data_fim)
        df = df_completo.loc[mask].reset_index(drop=True)
        
        if len(df) < 2:
            st.error("⚠️ O período selecionado precisa ter pelo menos 2 meses de dados para o cálculo de retorno.")
            st.stop()
            
        df['Retorno'] = df['Cota'].pct_change()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Retorno']).reset_index(drop=True)
        
        n_meses = len(df)
        anos = n_meses / 12
        retorno_medio_mensal = df['Retorno'].mean()
        volatilidade_mensal = df['Retorno'].std()
        
        retorno_anualizado = ((1 + retorno_medio_mensal) ** 12) - 1
        volatilidade_anualizada = volatilidade_mensal * np.sqrt(12)
        
        st.subheader("1. Resumo da Amostra")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Período Analisado", f"{formatar_decimal_br(anos)} anos")
        col2.metric("Meses (N)", f"{n_meses}")
        col3.metric("Retorno Médio (a.a.)", formatar_percentual_br(retorno_anualizado))
        col4.metric("Volatilidade (a.a.)", formatar_percentual_br(volatilidade_anualizada))
        
        st.divider()

        st.subheader("2. O Multiverso do Azar: Simulação de Monte Carlo")
        st.markdown(f"""
        O que aconteceria se pegarmos todos os retornos mensais do fundo gerados entre **{data_inicio_str} e {data_fim_str}** e sortearmos a ordem deles 10.000 vezes? 
        """)
        
        num_simulacoes = 10000
        retornos_historicos = df['Retorno'].values
        
        with st.spinner('Gerando 10.000 universos paralelos (Bootstrapping)...'):
            indices_aleatorios = np.random.randint(0, n_meses, size=(n_meses, num_simulacoes))
            simulacoes_retornos = retornos_historicos[indices_aleatorios]
            
            caminhos_acumulados = np.cumprod(simulacoes_retornos + 1, axis=0)
            caminhos_acumulados = np.vstack([np.ones(num_simulacoes), caminhos_acumulados]) 
            
            trajetoria_media = np.mean(caminhos_acumulados, axis=1)
            trajetoria_mediana = np.median(caminhos_acumulados, axis=1)
            
            limite_percentil = np.percentile(caminhos_acumulados[-1, :], 95.45)
            mascara_qualificados = caminhos_acumulados[-1, :] <= limite_percentil
            caminhos_qualificados = caminhos_acumulados[:, mascara_qualificados]
            trajetoria_media_qualificada = np.mean(caminhos_qualificados, axis=1)
            
            trajetoria_real = np.cumprod(retornos_historicos + 1)
            trajetoria_real = np.insert(trajetoria_real, 0, 1.0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            eixo_x = np.arange(n_meses + 1)
            
            ax.plot(eixo_x, caminhos_acumulados[:, :500], color='lightgray', alpha=0.15)
            
            ax.plot(eixo_x, trajetoria_media, color='#E67E22', linestyle='--', linewidth=2.5, label='Média Bruta (Distorcida por Outliers)')
            ax.plot(eixo_x, trajetoria_media_qualificada, color='#27AE60', linestyle='-.', linewidth=2.5, label='Média Qualificada (95,45% da Amostra)')
            ax.plot(eixo_x, trajetoria_mediana, color='#C0392B', linestyle=':', linewidth=2.5, label='Mediana (Cenário Base - P50)')
            ax.plot(eixo_x, trajetoria_real, color='#004488', linewidth=3, label='Trajetória Real do Fundo')
            
            teto_simulado = np.percentile(caminhos_acumulados[-1, :], 99)
            teto_real = np.max(trajetoria_real)
            limite_superior = max(teto_simulado, teto_real) * 1.05 
            
            ax.set_ylim(bottom=0, top=limite_superior)
            
            ax.set_title(f'Trajetória Real vs. {num_simulacoes} Caminhos Aleatórios', fontsize=14)
            ax.set_ylabel('Fator de Crescimento do Capital')
            ax.set_xlabel('Meses Alocados (No período selecionado)')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(x).replace(',', '.')))
            
        st.pyplot(fig)
        plt.close(fig) 
        
        st.markdown("""
        **A Leitura das Linhas:**
        * **Média Bruta (Laranja):** É fortemente distorcida para cima pela assimetria dos juros compostos. Alguns poucos universos onde a sorte foi extrema empurram essa métrica para fora da realidade.
        * **Média Qualificada (Verde):** Aqui removemos os universos de "ganho de loteria" (equivalente a cortar eventos acima de 2 desvios padrão). É um cenário-base muito mais justo e honesto para julgar a gestão.
        * **Mediana (Vermelha):** Representa exatamente o meio da amostra (P50). Se a linha azul do fundo orbita próxima à Mediana e à Média Qualificada, o gestor apenas capturou o prêmio de risco sistemático esperado para a sua volatilidade, sem evidências inquestionáveis de consistência fora do acaso.
        """)
        
    except Exception as e:
        st.error(f"Erro fatal: {e}")
