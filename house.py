import streamlit as st
import pickle

# Load the trained model
model = pickle.load(open('linreg.pkl', 'rb'))

# Load the scaler object
scaler = pickle.load(open('scaler.sav', 'rb'))


# Function to predict employee happiness
def predict_houseprice(area, beds, baths, stories, mainroad, guests, basement, heating, ac, parking, prefarea, furnishing):
    # Preprocess the input features if needed
    scaled = scaler.transform([[area, beds, baths, stories, mainroad, guests, basement, heating, ac, parking, prefarea, furnishing]])

    # Make the prediction using the trained model
    prediction = model.predict(scaled)

    return prediction[0]

# Streamlit app
def main():
    # Set the title and description of the app
    st.title('House Price Predictor')
    st.write('Enter the house details.')

    # Get user inputs
    area = st.number_input('Area')
    beds = st.number_input('Number of bedrooms', min_value=1, max_value=6, step=1)
    baths = st.number_input('Number of bathrooms', min_value=1, max_value=4, step=1)
    stories = st.number_input('Number of stories', min_value=1, max_value=4, step=1)
    mainroad = st.number_input('Proximity to main road (Yes:1, No:0)', min_value=0, max_value=1, step=1)
    guests = st.number_input('Number of guestrooms', min_value=0, max_value=1, step=1)
    basement = st.number_input('Is there a basement? (Yes:1, No:0)', min_value=0, max_value=1, step=1)
    heating = st.number_input('Is there hot water heating? (Yes:1, No:0)', min_value=0, max_value=1, step=1)
    ac = st.number_input('Is there air conditioning? (Yes:1, No:0)', min_value=0, max_value=1, step=1)
    parking = st.number_input('How many cars can be parked? (Yes:1, No:0)', min_value=0, max_value=3, step=1)
    prefarea = st.number_input('Prefarea', min_value=0, max_value=1, step=1)
    furnishing = st.number_input('Is the house furnished? (Furnished:0, Semi-furnished:1, Unfurnished:2)', min_value=0, max_value=2, step=1)

    # Make the prediction when the user clicks the 'Predict' button
    if st.button('Predict'):
        prediction = predict_houseprice(area, beds, baths, stories, mainroad, guests, basement, heating, ac, parking, prefarea, furnishing)
        st.write('Predicted House Price: ', '$ {:0,.0f}'.format(prediction))

# Run the app
if __name__ == '__main__':
    main()
