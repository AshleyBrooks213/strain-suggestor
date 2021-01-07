"""Machine learning functions"""

import logging
import random
import pickle

from fastapi import APIRouter, HTTPException
import pandas as pd
from pydantic import BaseModel, Field, validator


"""load NN Model"""
loaded_nn_model = pickle.load(open("app/pickled_nn_model", 'rb'))

"""load vectorizer"""
loaded_vectorizer = pickle.load(open("app/pickled_vectorizer", 'rb'))

"""read in csv file"""
df = pd.read_csv("app/cannabis.csv")


"""EDA"""
df['Flavor'] = df['Flavor'].str.replace('Blue,Cheese', 'Blue Cheese')
df = df[df['Effects'] != 'None']
df = df[df['Flavor'] != 'None']
df = df.set_index('Strain')
all_effects = df['Effects'].str.cat(sep=',')
set_effects = set(all_effects.split(","))
all_flavor = df['Flavor'].str.cat(sep=',')
set_flavor = set(all_flavor.split(","))
df['to_vect'] = df['Effects'] + "," + df['Flavor']
df.dropna(inplace=True)


log = logging.getLogger(__name__)
router = APIRouter()


"""List of Effects"""
effect_val_list = {
    'Aroused',
    'Creative',
    'Dry',
    'Energetic',
    'Euphoric',
    'Focused',
    'Giggly',
    'Happy',
    'Hungry',
    'Mouth',
    'Relaxed',
    'Sleepy',
    'Talkative',
    'Tingly',
    'Uplifted'}


"""List of Flavors"""
flavor_val_list = {
    'Ammonia',
    'Apple',
    'Apricot',
    'Berry',
    'Blue Cheese',
    'Blueberry',
    'Butter',
    'Cheese',
    'Chemical',
    'Chestnut',
    'Citrus',
    'Coffee',
    'Diesel',
    'Earthy',
    'Flowery',
    'Fruit',
    'Grape',
    'Grapefruit',
    'Honey',
    'Lavender',
    'Lemon',
    'Lime',
    'Mango',
    'Menthol',
    'Mint',
    'Minty',
    'Nutty',
    'Orange',
    'Peach',
    'Pear',
    'Pepper',
    'Pine',
    'Pineapple',
    'Plum',
    'Pungent',
    'Rose',
    'Sage',
    'Skunk',
    'Spicy/Herbal',
    'Strawberry',
    'Sweet',
    'Tar',
    'Tea',
    'Tobacco',
    'Tree',
    'Tropical',
    'Vanilla',
    'Violet',
    'Woody'}



class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    desired_effects: str = Field(..., example="Creative")
    desired_flavors: str = Field(..., example="Strawberry")
    

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])


@router.post('/predict')
async def predict(item: Item):
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `Effects`: str
    - `Flavor`: str
    
    ### Response
    - `prediction`: top 5 predictions based on user's input
    - `predict_proba`: float between 0.5 and 1.0, 
    representing the predicted class's probability

    Replace the placeholder docstring and fake predictions with your own model.
    """
    user_effect_list = item.desired_effects.split(',')
    user_flavor_list = item.desired_flavors.split(',')
    for effect in user_effect_list:
        if effect not in effect_val_list:
            raise HTTPException(status_code=404, detail=f'Effect {effect} not found')
    for flavor in user_flavor_list:
        if flavor not in flavor_val_list:
            raise HTTPException(status_code=404, detail=f'Flavor {flavor} not found')

    #need to combine the desired effects and flavors to match the fitting model
    desired_combined = item.desired_effects + ',' + item.desired_flavors

    #vectorize desired effects/flavors
    desired_dtm = loaded_vectorizer.transform([desired_combined])


    """Make predictions"""
    output = loaded_nn_model.kneighbors(desired_dtm.todense())

    """Show the nearest neighbors output"""
    nn = output[1][0] 
    df.iloc[nn].drop(columns='to_vect')

    return {
        'prediction': ", ".join(
            [df.iloc[nn].index[0],
            df.iloc[nn].index[1],
            df.iloc[nn].index[2],
            df.iloc[nn].index[3],
            df.iloc[nn].index[4]]),
    }



    
