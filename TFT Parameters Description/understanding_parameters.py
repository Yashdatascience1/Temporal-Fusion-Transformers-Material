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

columns = ['PARENT_DEALER_CODE', 'MODEL_FAMILY', 'PARENT_DEALER_CODE_MODEL_FAMILY',
       'BRAKE_TYPE', 'IGNITION_TYPE', 'WHEEL_TYPE', 'COLOUR', 'NET_SALES',
       'DUSSEHRA_(VIJAYADASHAMI)_DAYS', 'AKSHAYA_TRITIYA_DAYS',
       'BHAI_DOOJ_DAYS', 'BUDDHA_PURNIMA_DAYS', 'CHHATH_PUJA_DAYS',
       'DHANTERAS_DAYS', 'DIWALI_DAYS', 'EID_UL_FITR_DAYS',
       'GANESH_CHATURTHI_DAYS', 'GANGA_DUSSEHRA_DAYS', 'GOVARDHAN_POOJA_DAYS',
       'GURU_PURNIMA_DAYS', 'HANUMAN_JAYANTI_DAYS', 'HARTALIK_TEEJ_DAYS',
       'HOLI_DAYS', 'HOLIKA_DAHAN_DAYS', 'JAGANNATH_RATHYATRA_DAYS',
       'JANMASHTAMI_DAYS', 'KARWA_CHAUTH_DAYS', 'LOHRI_DAYS',
       'MAHA_SHIVARATRI_DAYS', 'MAKAR_SANKRANTI_PONGAL_DAYS',
       'NAG_PANCHAMI_DAYS', 'NAVRATRI_DAYS', 'NEW_YEAR_DAYS', 'ONAM_DAYS',
       'RAKSHA_BANDHAN_DAYS', 'REPUBLIC_DAY_DAYS', 'VASANT_PANCHAMI_DAYS',
       'VISHWAKARMA_PUJA_DAYS', 'MARRIAGE_DAYS', 'FESTIVE_PHASE_I',
       'FESTIVE_PHASE_II', 'FESTIVE_PHASE_III', 'PITRU_PAKSH',
       'PROP_FESTIVE_PHASE_I', 'PROP_FESTIVE_PHASE_II',
       'PROP_FESTIVE_PHASE_III', 'PROP_PITRU_PAKSH', 'DEALER_CITY',
       'X_CITY_CATEGORY', 'ZONAL_OFFICE_NAME','LAST_YEAR_CONTRIBUTION']


