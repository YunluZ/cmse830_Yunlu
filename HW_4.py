import streamlit as st
import time
from ucimlrepo import fetch_ucirepo
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

heart_disease = fetch_ucirepo(id=45)
df = heart_disease.data.original

st.set_option('deprecation.showPyplotGlobalUse', False)

col1, col2 = st.columns([2, 4])

col1.markdown('# Heart Disease Dataset')
col1.markdown('This app provides visualization of heart disease dataset.')
col1.markdown('_Hope you enjoy it :)_')

if "feedback" not in st.session_state:
    st.session_state["feedback"]="not done"


def change_fb_state():
    st.session_state["feedback"]="done"

with col1:
    feedback = st.text_input(
        "Provide us your feedback: 👇", on_change=change_fb_state
        )

prog_bar = col1.progress(0)

for per_completed in range(100):
    time.sleep(0.01)
    prog_bar.progress(per_completed+1)

col1.success("Feedback uploaded successfully!")

col1.metric(label="Tempreture", value="70 °F", delta="3 °F" )

if st.session_state["feedback"] == "done":
    if feedback:
        st.write("Thank you for your feedback. Your feedback is ", feedback)

col2.markdown("## Heart disease diagnosis based on sex")

with col2:
    df['num_group'] = (df['num'] > 0).astype(int)
    ax = sns.countplot(data=df, x='sex', hue='num_group', palette='Set2')

    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 1, f'{100 * height / total:.2f}%', ha="center", size = '10')

    plt.xticks([0, 1], ['Female', 'Male'])
    legend_labels = ['Not diagnosed', 'Diagnosed']
    plt.legend(labels=legend_labels)

    st.pyplot()

    with st.expander("Click to know more."):
        st.write("Here are more information of the possible risk factors associated with heart disease.")

        all_columns = df.columns[~(df.columns.isin(['num', 'num_group']))]
        num_rows = len(all_columns) // 2

        plt.figure(figsize=(20, 40))

        for i, column in enumerate(all_columns):
            plt.subplot(num_rows, 3, i + 1)
            sns.histplot(data=df, x=column, hue='num_group', element='step')
            plt.title(column)
            plt.tight_layout()
        st.pyplot()
