import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown("""
<style>
.main
{
background-color: #F5F5F5
}
</style>
""",unsafe_allow_html= True)

@st.cache
def get_data(filename):
    food_data = pd.read_csv(filename)

    return food_data

with header:
    st.title('Welcome to my awesom data science project!')
    st.text('in this project i look into the transactions of tacis in NYC. ...')

with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on blablabla.com. ...')

    food_data = get_data('ifood_df.csv')
    st.write(food_data.head())

    income_dist = pd.DataFrame(food_data['Income'].value_counts()).head(50)
    st.bar_chart(income_dist)

with features:
    st.header('The features I created')
    st.markdown('* **first features** i creat this feature becouses of this ')
    st.markdown('* **second features** i creat this feature becouses of this ')

with model_training:
    st.header('Time to train the model')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?',min_value=10, max_value =100, value=20, step=10)

    n_estimators = sel_col.selectbox('how many trees should there be ?', options=[100, 200, 300,'No Limit'], index=0)

    sel_col.text('Here is a list of features in my data:')
    sel_col.write(food_data.columns)


    input_feature = sel_col.text_input("Which feature should be used as the input feature?","Income")

    if n_estimators =='No Limit':

        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)




    X = (food_data[['Income']]).values
    y = (food_data[['AcceptedCmpOverall']]).values

    regr.fit(X,y)
    predicton = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is :')
    disp_col.write(mean_absolute_error(y, predicton))

    disp_col.subheader('Mean squared absolute error of the model is :')
    disp_col.write(mean_squared_error(y, predicton))

    disp_col.subheader('R squared error of the model is :')
    disp_col.write(r2_score(y, predicton))


