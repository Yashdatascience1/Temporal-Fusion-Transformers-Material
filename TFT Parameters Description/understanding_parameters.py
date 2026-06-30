from darts.dataprocessing.transformers import Scaler

# Function to encode the year as a normalized value
def encode_year(idx):
  return (idx.year - 2000) / 50

def encode_days_in_month(index):
  return index.days_in_month.to_numpy().reshape(-1,1)

# Set up the add_encoders dictionary to specify how different time-related encoders and transformers should be applied
add_encoders = {
    'cyclic': {'past': ['month'], 'future': ['month']},
    'position': {'past': ['relative'], 'future': ['relative']},
    'custom': {
        'past': [encode_year, encode_days_in_month],
        'future': [encode_year, encode_days_in_month]
    },
    'transformer': Scaler()
}



