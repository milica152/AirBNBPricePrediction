import sys
import numpy
import pandas as pd
from sklearn import model_selection, decomposition, preprocessing
from sklearn.decomposition import PCA


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
    data = data.drop('host_response_time', axis=1)    # 19 000 null values
    data = data.drop('host_response_rate', axis=1)    # 19 000 null values
    data = data.drop('host_acceptance_rate', axis=1)    # 19 000 null values
    data = data.drop('reviews_per_month', axis=1)   # 15 000 null values
    data = data.drop('security_deposit', axis=1)    # 15 000 null values
    data = data.drop('cleaning_fee', axis=1)    # 11 500 null values


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
    data = data.drop('amenities', axis=1)    # ZA SAD brisemo, kasnije mozda da obradimo da ima koristi


    # TEMPORARY DELETING
    data = data.drop('review_scores_rating', axis=1)
    data = data.drop('review_scores_accuracy', axis=1)
    data = data.drop('review_scores_cleanliness', axis=1)
    data = data.drop('review_scores_checkin', axis=1)
    data = data.drop('review_scores_communication', axis=1)
    data = data.drop('review_scores_location', axis=1)
    data = data.drop('review_scores_value', axis=1)

    return data


def fill_blank(data):
    #print(len(data))

    # delete rows if there are too few NA values in columns
    data = data.dropna(subset = ['host_is_superhost', 'host_listings_count', 'host_identity_verified', 'bathrooms', 'bedrooms', 'beds'])

    #print(len(data))

    return data


def encode_data(data):

    encoder = preprocessing.LabelEncoder()
    data['host_is_superhost'] = encoder.fit_transform(data['host_is_superhost'].astype('str'))
    data['host_identity_verified'] = encoder.fit_transform(data['host_identity_verified'].astype('str'))
    data['neighbourhood_cleansed'] = encoder.fit_transform(data['neighbourhood_cleansed'].astype('str'))
    data['is_location_exact'] = encoder.fit_transform(data['is_location_exact'].astype('str'))
    data['property_type'] = encoder.fit_transform(data['property_type'].astype('str'))
    data['room_type'] = encoder.fit_transform(data['room_type'].astype('str'))
    data['bed_type'] = encoder.fit_transform(data['bed_type'].astype('str'))
    data['has_availability'] = encoder.fit_transform(data['has_availability'].astype('str'))
    data['requires_license'] = encoder.fit_transform(data['requires_license'].astype('str'))
    data['instant_bookable'] = encoder.fit_transform(data['instant_bookable'].astype('str'))
    data['is_business_travel_ready'] = encoder.fit_transform(data['is_business_travel_ready'].astype('str'))
    data['cancellation_policy'] = encoder.fit_transform(data['cancellation_policy'].astype('str'))
    data['require_guest_profile_picture'] = encoder.fit_transform(data['require_guest_profile_picture'].astype('str'))
    data['require_guest_phone_verification'] = encoder.fit_transform(data['require_guest_phone_verification'].astype('str'))

    return data


if __name__ == '__main__':
    path = sys.argv[1]
    data = pd.read_csv(path)
    #print(data)

    data = drop_columns(data)
    #print(data)
    data = fill_blank(data)
    #print(data)
    data = encode_data(data)
    #print(data)

    x_train = data.drop('price', axis=1).values
    y_train = data['price'].values

    # Podelimo podatke na trening i test (20% je za testiranje)
    #train_x, test_x, train_y, test_y = \
    #    model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=2)

    # TRAIN

