### Need to only run once###
from CB_function import CB_Recommender
cb_recommender = CB_Recommender()

### Can change input employee ID and number of recommendations requested###
cb_recommender._rec_generator_('9442b3bf',5)