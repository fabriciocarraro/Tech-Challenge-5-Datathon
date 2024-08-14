#python -m venv DTGrupo56
#DTGrupo56/Scripts/activate
#streamlit run DTGrupo56.py
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import streamlit as st

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 10


# Usado para salvar os modelo treinado
from pickle4 import pickle


#Page config
def wide_space_default():
    st.set_page_config(layout='wide')


# ==============================================================================
wide_space_default()

# -------------------------------------------------------------------------------------------------------------------------------------------
#st.markdown("<h1 style='text-align: center; color: grey;'>Datathon - DTAT2</h1>", unsafe_allow_html=True)
st.title('Datathon - Passos Mágicos')
st.image('cabeçalho.jpg')
col1, col2 = st.columns(2)

with col1:
    st.subheader('Grupo 56')
    st.write(":point_right: Denise Oliveira     rm351364")
    st.write(":point_right: Fabrício Carraro    rm350902")
    st.write(":point_right: Luiz H. Spezzano    rm351120")
    st.write(":point_right: Thayze Darnieri     rm349021")

with col2:
    st.image("logo.png")


st.divider()  
st.write("**Desafio Proposto:**")
st.write("De acordo com os dados fornecidos pela Passos Mágicos, gerar análises e/ou desenvolver um modelo de Inteligência Artificial")

st.divider()  
st.write("**Solução Proposta:**")
st.write("Desenvolver um modelo de Inteligência Artificial que preveja se um aluno irá atingir o Ponto de Virada, de acordo com as seguintes variáveis:")
st.write(":black_medium_small_square: **PEDRA**")
st.write(":black_medium_small_square: **INDE** - Índice do Desenvolvimento Educacional, dado pela ponderação dos indicadores: IAN, IDA, IEG, IAA, IPS, IPP e IPV")
st.write(":black_medium_small_square: **IAN** - Indicador de Adequação ao Nível")
st.write(":black_medium_small_square: **IDA** - Indicador de Aprendizagem")
st.write(":black_medium_small_square: **IEG** - Indicador de Engajamento")
st.write(":black_medium_small_square: **IAA** - Indicador de Auto Avaliação")
st.write(":black_medium_small_square: **IPS** - Indicador Psicossocial")
st.write(":black_medium_small_square: **IPP** - Indicador Psicopedagógico")
st.write(":black_medium_small_square: **IPV** - Indicador de Ponto de Virada")
st.write("Adicionalmente disponibilizamos um painel de análises através deste :point_right: **[link](https://app.powerbi.com/view?r=eyJrIjoiOTExYmZiMTYtYmU5Mi00OTA5LThhNzAtMDEyOWM4NmFkZjJiIiwidCI6IjJlYzg1YTJlLTIzODEtNDZmMC1hM2RiLWQ5NDZkMmJhNDIyZiJ9)**")

st.divider()  
st.subheader('**Notas Iniciais do Grupo**')
st.write(":black_medium_small_square: Os dados foram previamente tratados e não demonstraremos aqui as técnicas utilizadas")
st.write(":black_medium_small_square: Para a varíavel Pedra, utlizamos a seguinte codificação: 1 - Quartzo / 2 - Ágata / 3 - Ametista / 4 - Topázio")
st.write(":black_medium_small_square: Para a varíavel Ponto de Virada, realizamos o encoding em booleano: 1 - Sim / 2 - Não ")

st.divider()  

tab1, tab2, tab3 = st.tabs(["Testar Modelos", "Treinar Novamente","Quadro Comparativo"])
with tab1:
    st.write("Preencha os campos abaixo e clique no botão 'Testar Modelo'") 
    with st.container():
        options = ["Quartzo", "Ágata", "Ametista","Topázio"]
        genre = st.radio(
            "**Selecione uma Pedra:**",
            options
        )
        n_Pedra = options.index(genre)+1
        col11, col12 = st.columns(2)
        with col11:        
            n_INDE = st.number_input("**INDE - Índice do Desenvolvimento Educacional**", min_value=0, max_value=10, value=None, placeholder="Digite um número de 0 a 10 ")
            n_IAN = st.number_input("**IAN - Indicador de Adequação ao Nível**", min_value=0, max_value=10, value=None, placeholder="Digite um número de 0 a 10 ")
            n_IDA = st.number_input("**IDA - Indicador de Aprendizagem**", min_value=0, max_value=10, value=None, placeholder="Digite um número de 0 a 10 ")
            n_IEG = st.number_input("**IEG - Indicador de Engajamento**", min_value=0, max_value=10, value=None, placeholder="Digite um número de 0 a 10 ")
        with col12:  
            n_IAA = st.number_input("**IAA - Indicador de Auto Avaliação**", min_value=0, max_value=10, value=None, placeholder="Digite um número de 0 a 10 ")
            n_IPS = st.number_input("**IPS - Indicador Psicossocial**", min_value=0, max_value=10, value=None, placeholder="Digite um número de 0 a 10 ")
            n_IPP = st.number_input("**IPP - Indicador Psicopedagógico**", min_value=0, max_value=10, value=None, placeholder="Digite um número de 0 a 10 ")
            n_IPV = st.number_input("**IPV - Indicador de Ponto de Virada**", min_value=0, max_value=10, value=None, placeholder="Digite um número de 0 a 10 ")

        col13, col14 = st.columns(2)
        with col13:    
            if st.button('Testar Modelo com KNN'):
                with open('modelo_KNN.pkl', 'rb') as f:
                    modelo_classificador  = pickle.load(f)
                    x = modelo_classificador.predict([[n_Pedra,n_INDE,n_IAN,n_IDA,n_IEG,n_IAA,n_IPS,n_IPP,n_IPV]])
                    
                    if x==1:
                        st.write(":sparkles: Com estes resultados, o Ponto de Virada será alcançado!")
                    else:
                        st.write(":x: Com estes resultados, o Ponto de Virada não será alcançado!")
        
        with col14: 
            if st.button('Testar Modelo com Random Forest'):
                with open('modelo_random_forest.pkl', 'rb') as f:
                    modelo_classificador_rf = pickle.load(f)
                    x = modelo_classificador_rf.predict([[n_Pedra,n_INDE,n_IAN,n_IDA,n_IEG,n_IAA,n_IPS,n_IPP,n_IPV]])
                    
                    if x==1:
                        st.write(":sparkles: Com estes resultados, o Ponto de Virada será alcançado!")
                    else:
                        st.write(":x: Com estes resultados, o Ponto de Virada não será alcançado!")



with tab2:
    st.write("Este projeto já trás dois modelos treinados prontos para uso. Caso queira/necessite retreinar os modelos, proceda da seguinte maneira:") 
    st.write(":black_medium_small_square: Faça o upload de um arquivo CSV com o nome 'DadosNormalizados-Final.csv' com a estrutura conforme abaixo")
    st.image("cabecalho_arquivo.png")
    st.write("Caso queira, faça o download do arquivo atual e proceda com os ajustes necessários.") 
    with open(arquivo_final, "rb") as file:
        btn = st.download_button(
            label="Download arquivo CSV atual",
            data=file,
            file_name=arquivo_final,
            mime="text/csv",
        )    

    st.write(":black_medium_small_square: Garantir que os números estão com '.' como separador decimal")
    st.write(":black_medium_small_square: Garantir que não existam valores nulos no arquivo")
    
    uploaded_file = st.file_uploader('Choose file',type=['csv','txt'])

    if uploaded_file:
        with open(arquivo_final, 'wb') as file:
            file.write(uploaded_file.getbuffer()) 
            st.write('Arquivo Salvo com sucesso!')
    
    col21, col22 = st.columns(2)
    with col21:
        st.subheader('Retreino KNN')
        knn_RS = st.number_input("**Random State**", value=77, placeholder="Informe um valor de Random State para treinar o modelo")
        knn_TS = st.number_input("**Test Size**", min_value=10, max_value=30, value=20, placeholder="Digite um número de 10 a 30 para definir o tamanho da base de testes")
        knn_NN = st.number_input("**N_Neighbors**", value=3, placeholder="Informe um valor de n_neighbors para treinar o modelo")

        if st.button('Retreinar o Modelo KNN'):
            df = pd.read_csv('DadosNormalizados-Final.csv', sep = ';')
            df.drop(['Ano', 'NOME','Idade'], axis=1, inplace=True)

            x = df[['PEDRA', 'INDE', 'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']]
            y = df['PONTO_VIRADA']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=knn_TS/100, stratify=y, random_state=knn_RS)

            modelo_classificador = KNeighborsClassifier(n_neighbors=knn_NN)
            modelo_classificador.fit(x_train, y_train)

            y_predito = modelo_classificador.predict(x_test)
            acc = accuracy_score(y_true = y_test, y_pred=y_predito) * 100
            precision = precision_score(y_test, y_predito) * 100
            recall = recall_score(y_test, y_predito) * 100
            f1 = f1_score(y_test,y_predito) * 100
            st.write("**Indicadores:**")
            st.write(f"Acuracia: {acc:.2f}%")
            st.write(f"Precisão: {precision:.2f}%")
            st.write(f"Recall: {recall:.2f}%")
            st.write(f"F1-Score: {f1:.2f}%")

            with open('modelo_KNN.pkl', 'wb') as file:
                pickle.dump(modelo_classificador, file)
            
            cm = confusion_matrix(y_test, y_predito)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            plt.xlabel("Valores preditos")
            plt.ylabel("Valores reais")
            plt.title("Confusion Matrix - KNN")
            st.pyplot(fig)

        # Plotando a curva ROC 
            fig1 = go.Figure()
            fig1.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1  )

            #fpr, tpr, _ = roc_curve(y_true, y_score)
            fpr, tpr, thresholds = roc_curve(y_test, y_predito)
            fig1.add_trace(go.Scatter(x=fpr, y=tpr,  mode='lines'))
            fig1.update_layout(
                title=f'Curva ROC (AUC={auc(fpr, tpr):.4f})',
                xaxis_title='Taxa de Falso Positivo',
                yaxis_title='Taxa de Verdadeiro Positivo',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500
            )
            st.plotly_chart(fig1, use_container_width=True)
#------------------------------------------------------------------------------------------------
    with col22:
        st.subheader('Retreino Random Forest')
        rf_RS = st.number_input("**Random State**", value=7, placeholder="Informe um valor de Random State para treinar o modelo")
        rf_TS = st.number_input("**RF Test Size**", min_value=10, max_value=30, value=20, placeholder="Digite um número de 10 a 30 para definir o tamanho da base de testes")
        rf_NE = st.number_input("**N_Estimators**", value=100, placeholder="Informe um valor de n_estimators para treinar o modelo")

        if st.button('Retreinar o Modelo Random Forest'):
            df = pd.read_csv('DadosNormalizados-Final.csv', sep = ';')
            df.drop(['Ano', 'NOME','Idade'], axis=1, inplace=True)

            x = df[['PEDRA', 'INDE', 'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']]
            y = df['PONTO_VIRADA']

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=rf_TS/100, random_state=rf_RS)

            modelo_classificador_rf = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(n_estimators=rf_NE, random_state=rf_RS))
            ])
            modelo_classificador_rf.fit(x_train, y_train)

            y_predito = modelo_classificador_rf.predict(x_test)
            acc = accuracy_score(y_true = y_test, y_pred=y_predito) * 100
            precision = precision_score(y_test, y_predito) * 100
            recall = recall_score(y_test, y_predito) * 100
            f1 = f1_score(y_test,y_predito) * 100
            st.write("**Indicadores:**")
            st.write(f"Acuracia: {acc:.2f}%")
            st.write(f"Precisão: {precision:.2f}%")
            st.write(f"Recall: {recall:.2f}%")
            st.write(f"F1-Score: {f1:.2f}%")

            with open('modelo_random_forest.pkl', 'wb') as file:
                pickle.dump(modelo_classificador_rf, file)

            cm = confusion_matrix(y_test, y_predito)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            plt.xlabel("Valores preditos")
            plt.ylabel("Valores reais")
            plt.title("Confusion Matrix")
            st.pyplot(fig)

            # Plotando a curva ROC 
            fig1 = go.Figure()
            fig1.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1  )

            #fpr, tpr, _ = roc_curve(y_true, y_score)
            fpr, tpr, thresholds = roc_curve(y_test, y_predito)
            fig1.add_trace(go.Scatter(x=fpr, y=tpr,  mode='lines'))
            fig1.update_layout(
                title=f'Curva ROC (AUC={auc(fpr, tpr):.4f})',
                xaxis_title='Taxa de Falso Positivo',
                yaxis_title='Taxa de Verdadeiro Positivo',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500
            )
            st.plotly_chart(fig1, use_container_width=True)

with tab3:
    st.write("Os resultados originais dos modelos treinados e testados neste projeto, seguem abaixo:") 
    col31, col32 = st.columns(2)

    with col31:
        st.subheader('KNN')
        st.write("Acurácia: 92.67%")
        st.write("Precisão: 84.09%")
        st.write("Recall: 58.73%")
        st.write("F1-Score: 69.16%")
        st.image("cm_knn.png")
        st.image("roc_KNN.PNG")
    with col32:
        st.subheader('Random Forest')
        st.write("Acurácia: 97.11%")
        st.write("Precisão: 90.00%")
        st.write("Recall: 88.52%")
        st.write("F1-Score: 89.26%")
        st.image("cm_RF.png")
        st.image("roc_RF.PNG")




st.divider()
st.write("Com base nos resultados demonstrados, a sugestão é seguir com o modelo Random Forest!") 
