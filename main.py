import sys
from tkinter import Image

import numpy as np
import pandas as pd
from pip._internal.req.req_file import preprocess
from scipy.stats import stats
from sklearn import model_selection, decomposition, preprocessing
from sklearn.decomposition import PCA
# from fancyimpute import MICE
import seaborn as sns
import matplotlib.pyplot as plt




def remove_outliers(data):
    z_scores = stats.zscore(data['price'])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    data = data[filtered_entries]
    return data

def feature_extraction(train_x):
    pca = decomposition.PCA(n_components=5)
    pca.fit(train_x)
    X_t_train = pca.transform(train_x)
    return X_t_train

def drop_columns(data):

    # deleting ids
    data = data.drop('id', axis=1)
    data = data.drop('scrape_id', axis=1)
    data = data.drop('host_id', axis=1)

    # deleting textual features
    data = data.drop('summary', axis=1)
    data = data.drop('space', axis=1)
    data = data.drop('description', axis=1)
    data = data.drop('name', axis=1)
    data = data.drop('notes', axis=1)
    data = data.drop('neighborhood_overview', axis=1)
    data = data.drop('access', axis=1)
    data = data.drop('interaction', axis=1)
    data = data.drop('transit', axis=1)
    data = data.drop('house_rules', axis=1)
    data = data.drop('host_name', axis=1)
    data = data.drop('host_location', axis=1)
    data = data.drop('host_about', axis=1)
    data = data.drop('host_neighbourhood', axis=1)
    data = data.drop('street', axis=1)

    # deleting urls
    data = data.drop('listing_url', axis=1)
    data = data.drop('thumbnail_url', axis=1)
    data = data.drop('medium_url', axis=1)
    data = data.drop('picture_url', axis=1)
    data = data.drop('xl_picture_url', axis=1)
    data = data.drop('host_url', axis=1)
    data = data.drop('host_thumbnail_url', axis=1)
    data = data.drop('host_picture_url', axis=1)

    # deleting dates
    data = data.drop('last_scraped', axis=1)
    data = data.drop('host_since', axis=1)
    data = data.drop('first_review', axis=1)
    data = data.drop('last_review', axis=1)
    data = data.drop('calendar_updated', axis=1)
    data = data.drop('calendar_last_scraped', axis=1)

    # deleting none values
    data = data.drop('experiences_offered', axis=1)
    data = data.drop('neighbourhood_group_cleansed', axis=1)
    data = data.drop('square_feet', axis=1)
    data = data.drop('weekly_price', axis=1)
    data = data.drop('monthly_price', axis=1)
    data = data.drop('license', axis=1)
    data = data.drop('jurisdiction_names', axis=1)
    data = data.drop('host_response_time', axis=1)    # 19 310 null values (vise od pola)
    data = data.drop('host_response_rate', axis=1)    # 19 310 null values (vise od pola)
    data = data.drop('host_acceptance_rate', axis=1)    # 12 643 null values

    # deleting redundant values
    data = data.drop('host_total_listings_count', axis=1)
    data = data.drop('neighbourhood', axis=1)
    data = data.drop('calculated_host_listings_count', axis=1)

    # deleting numeric values
    data = data.drop('minimum_nights', axis=1)
    data = data.drop('maximum_nights', axis=1)
    data = data.drop('minimum_minimum_nights', axis=1)
    data = data.drop('maximum_minimum_nights', axis=1)
    data = data.drop('minimum_maximum_nights', axis=1)
    data = data.drop('maximum_maximum_nights', axis=1)

    # deleting other irrelevant
    data = data.drop('host_has_profile_pic', axis=1)
    data = data.drop('city', axis=1)
    data = data.drop('state', axis=1)
    data = data.drop('zipcode', axis=1)
    data = data.drop('market', axis=1)
    data = data.drop('smart_location', axis=1)
    data = data.drop('country_code', axis=1)
    data = data.drop('country', axis=1)

    return data


def fill_blank(data):
    # delete rows if there are too few NA values in columns
    data = data.dropna(subset = ['host_is_superhost', 'host_listings_count', 'host_identity_verified', 'bathrooms', 'bedrooms', 'beds',
                                 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                                 'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                                 'review_scores_value'])

    # security_deposit
    val_to_fill_with_sd = data['security_deposit'].dropna(axis=0).str.strip('$').str.replace(',', '').astype('float').median()
    data['security_deposit'] = data['security_deposit'].fillna(str(val_to_fill_with_sd)).str.strip('$').str.replace(',', '').astype('float')

    # cleaning_fee
    val_to_fill_with_cf = data['cleaning_fee'].dropna(axis=0).str.strip('$').str.replace(',', '').astype('float').median()
    data['cleaning_fee'] = data['cleaning_fee'].fillna(str(val_to_fill_with_cf)).str.strip('$').str.replace(',', '').astype('float')

    #reviews_per_month
    data['reviews_per_month'] = data['reviews_per_month'].fillna(data['reviews_per_month'].median())

    return data


def encode_data(data):
    # ENCODING

    # dummies neighbourhood_cleansed
    dummies = pd.get_dummies(data['neighbourhood_cleansed'], prefix='neighbourhood')
    data = data.drop('neighbourhood_cleansed', axis=1)
    data = data.join(dummies)

    # dummies property_type
    dummies = pd.get_dummies(data['property_type'], prefix='property_type')
    data = data.drop('property_type', axis=1)
    data = data.join(dummies)


    # dummies cancellation_policy
    dummies = pd.get_dummies(data['cancellation_policy'], prefix='
                             
                             ')
    data = data.drop('cancellation_policy', axis=1)
    data = data.join(dummies)


    # dummies room_type
    dummies = pd.get_dummies(data['room_type'], prefix='room_type')
    data = data.drop('room_type', axis=1)
    data = data.join(dummies)


    # dummies bed_type
    dummies = pd.get_dummies(data['bed_type'], prefix='bed_type')
    data = data.drop('bed_type', axis=1)
    data = data.join(dummies)

    return data


def transform_data(data):
    #remove % from fields and convert to float
    # data['host_acceptance_rate'] = data['host_acceptance_rate'].str.replace('%', '').astype(float)

    #boolean vrednosti
    bool_col = ['host_is_superhost', 'host_identity_verified', 'is_location_exact',
                'has_availability', 'requires_license', 'instant_bookable', 'is_business_travel_ready',
                'require_guest_profile_picture', 'require_guest_phone_verification']
    data[bool_col] = data[bool_col].replace({'t': 1, 'f': 0})

    #prices remove $ and convert to float
    #dodaj 'security_deposit', 'cleaning_fee' kad se fill NA
    data['price'] = data['price'].str.replace('$', '')
    data['price'] = data['price'].str.replace(',', '').astype(float)
    data['extra_people'] = data['extra_people'].str.replace('$', '')
    data['extra_people'] = data['extra_people'].str.replace(',', '').astype(float)

    #prebrojimo koliko koji host ima nacina verifikacije
    data.host_verifications = data.host_verifications.str.replace("[", "")
    data.host_verifications = data.host_verifications.str.replace("]", "")
    data.host_verifications = data.host_verifications.str.replace("'", "")
    data['host_verifications_count'] = data.apply(lambda row: row.host_verifications.count(',') + 1 if not type(row) is float else 0,axis=1)
    data = data.drop('host_verifications', axis=1)

    # prebrojimo koliko koji oglas ima dodatnih usluga
    data['amenities_count'] = data.apply(lambda row: row.amenities.count(',') + 1 if not type(row) is float else 0, axis=1)
    data = data.drop('amenities', axis=1)

    #ako je broj reviewa 0 onda review_scores popunimo sa 0 (one koje su null)
    data['review_scores_rating'] = np.where((data['number_of_reviews'] == 0) & (data['review_scores_rating'].isnull()), 0, data.review_scores_rating)
    data['review_scores_accuracy'] = np.where((data['number_of_reviews'] == 0) & (data['review_scores_accuracy'].isnull()), 0, data.review_scores_accuracy)
    data['review_scores_cleanliness'] = np.where((data['number_of_reviews'] == 0) & (data['review_scores_cleanliness'].isnull()), 0, data.review_scores_cleanliness)
    data['review_scores_checkin'] = np.where((data['number_of_reviews'] == 0) & (data['review_scores_checkin'].isnull()), 0, data.review_scores_checkin)
    data['review_scores_communication'] = np.where((data['number_of_reviews'] == 0) & (data['review_scores_communication'].isnull()), 0, data.review_scores_communication)
    data['review_scores_location'] = np.where((data['number_of_reviews'] == 0) & (data['review_scores_location'].isnull()), 0, data.review_scores_location)
    data['review_scores_value'] = np.where((data['number_of_reviews'] == 0) & (data['review_scores_value'].isnull()), 0, data.review_scores_value)

    return data


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    path = sys.argv[1]
    data = pd.read_csv(path)
    #print(data)

    data = drop_columns(data)
    data = transform_data(data)
    data = fill_blank(data)
    data = encode_data(data)
    data = remove_outliers(data)

    # trans = MICE(verbose=False)
    # f_complete = trans.complete(data)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #
    #     print(data.size())

    x_train = data.drop('price', axis=1).values
    y_train = data['price'].values

    # Podelimo podatke na trening i test (20% je za testiranje)
    #train_x, test_x, train_y, test_y = \
    #    model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=2)

    # TRAIN

