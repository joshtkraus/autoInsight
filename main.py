import pandas as pd
from flask import Flask, request, jsonify, render_template, session
import csv
from scipy import signal
import pickle

app = Flask(__name__)
app.secret_key = 'BOu3ndKPDWaNL2orvl5vepT4'

# Drop Down CSV
# Load the CSV data
data = []
with open("static/dropdown.csv", 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        data.append(row)
# Extract unique Make and model values for default
make_options = sorted(set(row['Make'] for row in data))

# Load Model
loaded_model = pickle.load(open('static/gbr_pkl.sav', 'rb'))

# cols
cols = ['msrp',
 'days_on_lot',
 'mileage',
 'vf_DisplacementCC',
 'vf_ModelYear',
 'isNew_True',
 'brandName_ALFA ROMEO',
 'brandName_ASTON MARTIN',
 'brandName_AUDI',
 'brandName_BENTLEY',
 'brandName_BMW',
 'brandName_BUICK',
 'brandName_CADILLAC',
 'brandName_CHEVROLET',
 'brandName_CHRYSLER',
 'brandName_DATSUN',
 'brandName_DODGE',
 'brandName_FIAT',
 'brandName_FISKER AUTOMOTIVE',
 'brandName_FORD',
 'brandName_GENESIS',
 'brandName_GMC',
 'brandName_HONDA',
 'brandName_HUMMER',
 'brandName_HYUNDAI',
 'brandName_INFINITI',
 'brandName_ISUZU',
 'brandName_JAGUAR',
 'brandName_JEEP',
 'brandName_KIA',
 'brandName_LAND ROVER',
 'brandName_LEXUS',
 'brandName_LINCOLN',
 'brandName_LOTUS',
 'brandName_MASERATI',
 'brandName_MAZDA',
 'brandName_MCLAREN',
 'brandName_MERCEDES-BENZ',
 'brandName_MERCURY',
 'brandName_MINI',
 'brandName_MITSUBISHI',
 'brandName_NISSAN',
 'brandName_OLDSMOBILE',
 'brandName_PLYMOUTH',
 'brandName_PONTIAC',
 'brandName_PORSCHE',
 'brandName_RAM',
 'brandName_ROLLS ROYCE',
 'brandName_SAAB',
 'brandName_SATURN',
 'brandName_SMART',
 'brandName_SPRINTER (DODGE OR FREIGHTLINER)',
 'brandName_SUBARU',
 'brandName_SUZUKI',
 'brandName_TOYOTA',
 'brandName_VOLKSWAGEN',
 'brandName_VOLVO']

# Prediction Values
# vals
msrp = [47500]
days_on_lot = [183]
mileage = [75000]
vf_DisplacementCC = [4000]
vf_ModelYear = [2021]
brand = 'Acura'
isNew = "New"
# store
vals = {'msrp':msrp,'days_on_lot':days_on_lot,'mileage':mileage,'vf_DisplacementCC':vf_DisplacementCC,'vf_ModelYear':vf_ModelYear}
# check condition
if isNew == "New":
    vals['isNew_True'] = True
# get brand col name
brand_col = 'brandName_'+brand.upper()
# fill rest of cols with false
for c in cols:
    if c == brand_col:
        vals[c] = True
    else:
        if c not in vals.keys():
            vals[c] = False
# to df
pred_vals = pd.DataFrame.from_dict(vals)
# Predict
predicted_value = round(loaded_model.predict(pred_vals)[0])

# Initialize Graph
# point prediction
x_value = 47500
# x axis values
x = list(range(15000,80000,1000))
mid = []
for msrp in range(15000,80000,1000):
    vals['msrp'] = [msrp]
    df = pd.DataFrame.from_dict(vals)
    # Predict
    pred = round(loaded_model.predict(df)[0])
    mid.append(pred)
# upper and lower
sd = 2473.4417250378865
z = 1.96
upper = [val + z * sd for val in mid]
lower = [val - z * sd for val in mid]
# point prediciton
idx = x.index(x[min(range(len(x)), key = lambda i: abs(x[i]-x_value))])
upper_point = upper[idx]
lower_point = lower[idx]
# smoothing
x=signal.savgol_filter(x, 10, 1).tolist()
mid=signal.savgol_filter(mid, 10, 1).tolist()
upper=signal.savgol_filter(upper, 10, 1,).tolist()
lower=signal.savgol_filter(lower, 10, 1).tolist()
# offer 
offer_value=40000

# x axis selection
x_axis = 'MSRPChart'

# render default dropdown values
@app.route('/')
def home():
    return render_template('dashboard.html',
                           make_options=make_options,
                            x=x, mid=mid, upper=upper, lower=lower,
                          x_value = x_value, predicted_value=predicted_value, offer_value=offer_value,
                           upper_point=upper_point, lower_point=lower_point, x_axis = x_axis)

@app.route('/predict',methods=['POST'])
def predict():
    
    # get responses
    int_features = [x for x in request.form.values()]

    # values
    # numeric
    msrp = [int(int_features[6])]
    days_on_lot = [int(int_features[5])]
    vf_DisplacementCC = [int(int_features[4])]
    vf_ModelYear = [int(int_features[1])]
    mileage = [int(int_features[3])]
    isNew = int_features[0]
    brand = int_features[2]
   # store
    vals = {'msrp':msrp,'days_on_lot':days_on_lot,'mileage':mileage,'vf_DisplacementCC':vf_DisplacementCC,'vf_ModelYear':vf_ModelYear}
    # check condition
    if isNew == "New":
        vals['isNew_True'] = True
    # get brand col name
    brand_col = 'brandName_'+brand.upper()
    # fill rest of cols with false
    for c in cols:
        if c == brand_col:
            vals[c] = True
        else:
            if c not in vals.keys():
                vals[c] = False
    # to df
    pred_vals = pd.DataFrame.from_dict(vals)
    # Predict
    predicted_value = round(loaded_model.predict(pred_vals)[0])

    # Create X & Y Values for Viz
    if (int_features[10] == 'MSRPChart'):
        x_value = int(int_features[6])
        x = list(range(15000,80000,1000))
        mid = []
        for msrp in range(15000,80000,1000):
            vals['msrp'] = [msrp]
            df = pd.DataFrame.from_dict(vals)
            # Predict
            pred = round(loaded_model.predict(df)[0])
            mid.append(pred)
    elif (int_features[10] == "DaysChart"):
        x_value = int(int_features[5])
        x = list(range(0,366,5))
        mid = []
        for days in range(0,366,5):
            vals['days_on_lot'] = [days]
            df = pd.DataFrame.from_dict(vals)
            # Predict
            pred = round(loaded_model.predict(df)[0])
            mid.append(pred)
    elif (int_features[10] == "EngineChart"):
        x_value = int(int_features[4])
        x = list(range(1000,7000,100))
        mid = []
        for engine in range(1000,7000,100):
            vals['vf_DisplacementCC'] = [engine]
            df = pd.DataFrame.from_dict(vals)
            # Predict
            pred = round(loaded_model.predict(df)[0])
            mid.append(pred)
    elif (int_features[10] == 'YearChart'):
        x_value = int(int_features[1])
        x = list(range(2000,2024,1))
        mid = []
        for year in range(2000,2024,1):
            vals['vf_ModelYear'] = [year]
            df = pd.DataFrame.from_dict(vals)
            # Predict
            pred = round(loaded_model.predict(df)[0])
            mid.append(pred)
    else:
        x_value = int(int_features[3])
        x = list(range(0,150000,1000))
        mid = []
        for mileage in range(0,150000,1000):
            vals['mileage'] = [mileage]
            df = pd.DataFrame.from_dict(vals)
            # Predict
            pred = round(loaded_model.predict(df)[0])
            mid.append(pred)

    # Create Upper and Lower Bounds
    if (int_features[9] == 'Conservative'):
        z = 1.28
        upper = [val + z * sd for val in mid]
        lower = [val - z * sd for val in mid]
        # point prediciton
        idx = x.index(x[min(range(len(x)), key = lambda i: abs(x[i]-x_value))])
        upper_point = upper[idx]
        lower_point = lower[idx]
    elif (int_features[9] == 'Moderate'):
        z = 1.645
        upper = [val + z * sd for val in mid]
        lower = [val - z * sd for val in mid]   
        # point prediciton
        idx = x.index(x[min(range(len(x)), key = lambda i: abs(x[i]-x_value))])
        upper_point = upper[idx]
        lower_point = lower[idx]
    else:
        z = 1.96
        upper = [val + z * sd for val in mid]
        lower = [val - z * sd for val in mid]
        # point prediciton
        idx = x.index(x[min(range(len(x)), key = lambda i: abs(x[i]-x_value))])
        upper_point = upper[idx]
        lower_point = lower[idx]

    # smoothing
    if (int_features[10] == 'MSRPChart'):
        # smoothing
        mid=signal.savgol_filter(mid, 10, 1).tolist()
        upper=signal.savgol_filter(upper, 10, 1,).tolist()
        lower=signal.savgol_filter(lower, 10, 1).tolist()
    elif (int_features[10] == "DaysChart"):
        # smoothing
        mid=signal.savgol_filter(mid, 10, 1).tolist()
        upper=signal.savgol_filter(upper, 10, 1,).tolist()
        lower=signal.savgol_filter(lower, 10, 1).tolist()
    elif (int_features[10] == "EngineChart"):
        # smoothing
        mid=signal.savgol_filter(mid, 25, 1).tolist()
        upper=signal.savgol_filter(upper, 25, 1,).tolist()
        lower=signal.savgol_filter(lower, 25, 1).tolist()
    elif (int_features[10] == "YearChart"):
        # smoothing
        mid=signal.savgol_filter(mid, 5, 1).tolist()
        upper=signal.savgol_filter(upper, 5, 1,).tolist()
        lower=signal.savgol_filter(lower, 5, 1).tolist()
    else:
        # smoothing
        mid=signal.savgol_filter(mid, 50, 1).tolist()
        upper=signal.savgol_filter(upper, 50, 1,).tolist()
        lower=signal.savgol_filter(lower, 50, 1).tolist()

    # set lower bound
    mid = [val if val > 0 else 0 for val in mid]
    upper = [val if val > 0 else 0 for val in upper]
    lower = [val if val > 0 else 0 for val in lower]

    # Get values
    if request.method == 'POST':

        selected_value_msrp = request.form['MSRP']
        session['selected_value_msrp'] = selected_value_msrp
  
        selected_value_days = request.form['Days']
        session['selected_value_days'] = selected_value_days

        selected_value_engine = request.form['Engine']
        session['selected_value_engine'] = selected_value_engine

        selected_value_new = request.form['New']
        session['selected_value_new'] = selected_value_new

        selected_value_year = request.form['Year']
        session['selected_value_year'] = selected_value_year

        selected_value_mileage = request.form['Mileage']
        session['selected_value_mileage'] = selected_value_mileage

        selected_value_make = request.form['Make']
        session['selected_value_make'] = selected_value_make          

        selected_value_offer = request.form['Offer Price']
        session['selected_value_offer'] = selected_value_offer    
 
        selected_value_strategy = request.form['Offer Strategy']
        session['selected_value_strategy'] = selected_value_strategy  

        selected_value_buysell = request.form['Buy Sell']
        session['selected_value_buysell'] = selected_value_buysell  

    return render_template('dashboard.html',
                            make_options=make_options,
                          x=x, mid=mid, upper=upper, lower=lower,
                          x_value = x_value, predicted_value=predicted_value, offer_value=selected_value_offer,
                           upper_point=upper_point, lower_point=lower_point, x_axis = str(int_features[10]),
                           selected_value_year=selected_value_year, selected_value_mileage = selected_value_mileage, selected_value_msrp = selected_value_msrp, \
                           selected_value_make = selected_value_make, selected_value_engine = selected_value_engine, selected_value_offer = selected_value_offer, \
                           selected_value_days = selected_value_days, selected_value_new = selected_value_new, \
                           selected_value_strategy = selected_value_strategy, selected_value_buysell=selected_value_buysell)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    int_features = list(data.values())
    
    # values
    # numeric
    msrp = [int(int_features[6])]
    days_on_lot = [int(int_features[5])]
    vf_DisplacementCC = [int(int_features[4])]
    vf_ModelYear = [int(int_features[1])]
    mileage = [int(int_features[3])]
    isNew = int_features[0]
    brand = int_features[2]
   # store
    vals = {'msrp':msrp,'days_on_lot':days_on_lot,'mileage':mileage,'vf_DisplacementCC':vf_DisplacementCC,'vf_ModelYear':vf_ModelYear}
    # check brand name
    if brand != "Acura":
        # store
        vals['brandName_'+brand.upper()] = True
    # check condition
    if isNew == "New":
        vals['isNew_True'] = True
    # fill rest of cols with false
    for c in cols:
        if c not in vals.keys():
            vals[c] = False
    # to df
    pred_vals = pd.DataFrame.from_dict(vals)
    # Predict
    predicted_value = round(loaded_model.predict(pred_vals)[0])

    return jsonify(predicted_value)

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=3000, debug=True)