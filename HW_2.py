import streamlit as st
import seaborn as sns
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


df_iris = sns.load_dataset("iris")
df_mpg = sns.load_dataset('mpg')

fig = px.scatter_3d(
    df_iris,
    x="sepal_length",
    y="sepal_width",
	z='petal_length',
    color="species",
)

fig_1 = px.scatter_3d(
    df_iris.query("petal_width <= 1.3"),
    x="sepal_length",
    y="sepal_width",
	z='petal_length',
    color="species",
)

fig_2 = px.scatter_3d(
    df_iris.query("petal_width > 1.3"),
    x="sepal_length",
    y="sepal_width",
	z='petal_length',
    color="species",
)

tab1, tab2, tab3, tab4= st.tabs(['Iris dataset', "Petal width less than 1.3", "Petal width larger than 1.3", 'Bar chart using matplotlib'])
with tab1:
	st.plotly_chart(fig, theme=None, use_container_width=True)

with tab2:
    st.plotly_chart(fig_1, theme=None, use_container_width=True)

with tab3:
    st.plotly_chart(fig_2, theme=None, use_container_width=True)

with tab4:
	fig_41, ax = plt.subplots()
	ax.bar(df_mpg['origin'], df_mpg['horsepower'], color='pink')
	ax.set_xlabel('Origin')
	ax.set_ylabel('Horsepower')
	plt.title('Housepower of cars from different origin')
	st.pyplot(fig_41)
