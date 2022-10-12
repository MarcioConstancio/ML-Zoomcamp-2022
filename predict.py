import pickle

model_file = "model1.bin"
dv_file = 'dv.bin'

with open(model_file, 'rb' ) as f_in:
	model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in: 
    dv = pickle.load(f_in)

customer = {
	"reports": 0,
	"share": 0.001694,
	"expenditure": 0.12,
	"owner": "yes"
	}

X = dv.transform([customer])

y_prob = model.predict_proba(X)[0,1]

print('input: ',customer)
print('probability of getting a credit card: ',round(y_prob,3))