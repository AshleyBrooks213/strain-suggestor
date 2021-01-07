# Unit 3 Build Week Project: Strain-Suggestor

The purpose of this project is to deploy an app on Heroku and have a model that returns the top predicted strain of marijuana based on a user's input.
Our model will return the top 5 predictions for suggest strains based on user's input of desired effects and flavors.

You can find the finished product here:
https://strain-suggestor.herokuapp.com/

You can find the completed models and vectorized files in the directory included or in the github repo here:
https://github.com/micahks/Med-Cabinet/tree/master/Med%20Cabinet


This project was completed using the FastAPI framework.

## Install 
How to install all dependencies, launch the app and run it locally.

Windows:

```sh
pipenv install --dev
```

Activate Virtual Environment:

```sh
pipenv shell
```

Launch App:

```sh
uvicorn app.main:app --reload
```

Browser:

```sh
localhost:8000
```
