import streamlit as st
from multiapp import MultiApp
# from apps import home, model_random_forest, model_clustering, model_svm, model_regresion_logistica, model_lib_prophet # import your app modules here

from apps import home, model_arima, model_random_forest
app = MultiApp()

st.markdown("""
#  Equipo C - Modelos 
""")

# Add all your application here
app.add_app("Home", home.app)
# app.add_app("Modelo Random Forest", model_random_forest.app)
app.add_app("Modelo Arima", model_arima.app)
app.add_app("Decision tree", model_random_forest.app)
# app.add_app("Modelo Regresión Logística", model_regresion_logistica.app)
# app.add_app("Modelo SVM de Regresión", model_svm.app)
# app.add_app("Modelo basado en la librería Prophet", model_lib_prophet.app)
# app.add_app("Caso de asociación clustering", model_clustering.app)
# The main app
app.run()