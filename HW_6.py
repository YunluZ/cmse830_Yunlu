import streamlit as st
import time
from ucimlrepo import fetch_ucirepo
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)

heart_disease = fetch_ucirepo(id=45)
df = heart_disease.data.original
df['num_group'] = (df['num'] > 0).astype(int)

def diag(num):
    if num == 0:
        return 'Not Diagnosed'
    elif num == 1:
        return 'Diagnosed'
    else:
        return 'Unknown'

def gen(num):
    if num == 0:
        return 'Female'
    elif num == 1:
        return 'Male'
    else:
        return 'Unknown'

df['diagnosis'] = df['num_group'].apply(diag)
df['gender'] = df['sex'].apply(gen)

df = df.drop(columns=['sex', 'num', 'num_group'])

st.markdown('# Heart Disease')
st.markdown('Heart disease is a global health concern, standing as one of the primary contributors to worldwide mortality rates.')
st.markdown('UCI Heart Disease dataset emerges as a highly promising resource for evaluating heart disease risk.')
st.markdown('This web app provides visualization of heart disease dataset, which is designed for users to perform targeted searches,\
gain profound insights,and engage in comprehensive data analysis customized to their individual health profiles.')

tab1, tab2= st.tabs(['Diagnosis Distribution', 'Correlation'])

with tab1:
    st.markdown('### The distribution of diagnosis results based on different risk factors.')
    st.markdown('UCI Heart Disease dataset comprises a wealth of patient information,\
     spanning age, gender, cholesterol levels, blood pressure, and various other crucial indicators.')


# Define a dictionary with options and corresponding plotting logic
    plot_options = {
        "age": lambda: sns.histplot(data=df, x="age", hue='diagnosis', element='step'),
        "gender": lambda: sns.histplot(data=df, x="gender", hue='diagnosis', element='step'),
        "cp": lambda: sns.histplot(data=df, x="cp", hue='diagnosis', element='step'),
        "trestbps": lambda: sns.histplot(data=df, x="trestbps", hue='diagnosis', element='step'),
        "chol": lambda: sns.histplot(data=df, x="chol", hue='diagnosis', element='step'),
        "fbs": lambda: sns.histplot(data=df, x="fbs", hue='diagnosis', element='step'),
        "restecg": lambda: sns.histplot(data=df, x="restecg", hue='diagnosis', element='step'),
        "thalach": lambda: sns.histplot(data=df, x="thalach", hue='diagnosis', element='step'),
        "exang": lambda: sns.histplot(data=df, x="exang", hue='diagnosis', element='step'),
        "oldpeak": lambda: sns.histplot(data=df, x="oldpeak", hue='diagnosis', element='step'),
        "slope": lambda: sns.histplot(data=df, x="slope", hue='diagnosis', element='step'),
        "cal": lambda: sns.histplot(data=df, x="cal", hue='diagnosis', element='step'),
        "thal": lambda: sns.histplot(data=df, x="thal", hue='diagnosis', element='step'),
        }

    selected_option = st.selectbox("Select the risk factor that you are interested in:", list(plot_options.keys()))

# Check if the selected option exists in the dictionary
    if selected_option in plot_options:
    # Call the corresponding plotting function
        plot_function = plot_options[selected_option]
        plot_function()
        plt.xlabel(selected_option)
        plt.ylabel('Density')
        st.pyplot()
    else:
        st.write("Select a valid option.")

    with st.expander("Click to know more."):
        st.write("### _Here are some information of the possible risk factors associated with heart disease._")
        st.markdown('1. The rate of heart disease diagnosis is much **higher** in **males** than in **females**.')
        st.markdown('2. The **age** of individuals with heart disease is generally **higher** compared to those without heart disease.')
        st.markdown('3. Most diagnosed individuals experience exercise-induced **angina** (exang) and **chest pain**).')
        st.markdown('4. Almost no undiagnosed individuals exhibit **ST depression** induced by exercise relative to rest.')
        st.markdown('5. Among diagnosed individuals, most have a **flat ST depression** induced by exercise relative to rest.')
        st.markdown('6. Individuals with **reversible thalassemia** tend to have a higher likelihood of heart disease.')


with tab2:
    st.markdown('### The relationship between two risk factors.')
    st.markdown('It is also important to know the relationship between risk factors.\
    Understanding the relationship could help us to prepare for the emerging symptoms.')

    col_list = ['age',
                'cp',
                'trestbps',
                'chol',
                'fbs',
                'restecg',
                'thalach',
                'exang',
                'oldpeak',
                'slope',
                'ca',
                'thal'
                ]

    st.markdown("**Select TWO risk factors that you are interested in:**")
    x_variable = st.selectbox("X Variable", col_list)
    y_variable = st.selectbox("Y Variable", col_list)

    # Create a scatter plot
    ax = sns.jointplot(data=df, x=x_variable, y=y_variable, hue='diagnosis')
    st.pyplot(ax.figure)

    df_n = df.drop(columns=['gender', 'diagnosis'])

    with st.expander("Correlation Heatmap"):
        sns.heatmap(df_n.corr(),
            vmin=-1,
            vmax=1,
            center=0,
            cmap='coolwarm'
            )
        plt.legend().remove()
        st.pyplot()
