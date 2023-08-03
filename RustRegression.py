import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

#import matplotlib.pyplot as plt
#from sklearn import tree as sktree


data = []


relevant_features = ["sttic", "rfn__", "ifn__", "rbb__", "ibb__", "rssd_", "issd_", "rstmt", "istmt", "rplac", "iplac", "rproj", "iproj", "rcnst", "icnst", "rdecl", "idecl"]
irrelevant_features = ['est__', 't_all__', 't_gen__', 't_opt__', 't_lto__']

for fname in ["Debug-Top1000-1.txt", "Opt-Top1000-1.txt","Debug-Top1000-2.txt", "Opt-Top1000-2.txt","Debug-Top1000-3.txt", "Opt-Top1000-3.txt"]:
    df = pd.read_csv(fname, sep = '\s+', header = 37, names = ["cgu_name", "sttic", "rfn__", "ifn__", "rbb__", "ibb__", "rssd_", "issd_", "rstmt", "istmt", "rplac", "iplac", "rproj", "iproj", "rcnst", "icnst", "rdecl", "idecl", "est__", "t_all__", "t_gen__", "t_opt__", "t_lto__"])
    
    del df['cgu_name']
    data.append(df)

def params_via_cv(data):
    param_grid = {
        'ccp_alpha': [ 0.1, 0.2, 0.3, 0.4],
        'min_samples_split': [2, 3, 4],
        'max_features': [2, 4, 6]
        
    }
    grid_regr = GridSearchCV( RandomForestRegressor(), param_grid, cv=5)
    grid_regr.fit(data[relevant_features], data["t_all__"])
    print(grid_regr.best_params_)



def analyze(data, rf):
    train_data, test_data, train_target, test_target = train_test_split(data[relevant_features], data["t_all__"],test_size=.3)
    
    rf.fit(train_data,train_target)
    
    feature_list = [(j,i) for (i,j) in zip(rf.feature_importances_, ["sttic", "rfn__", "ifn__", "rbb__", "ibb__", "rssd_", "issd_", "rstmt", "istmt", "rplac", "iplac", "rproj", "iproj", "rcnst", "icnst", "rdecl", "idecl"])]
    feature_list.sort(key=lambda e: e[1], reverse= True)
    for (name, importance) in feature_list:
        print(f"{name} {importance:.4f}")
        
    accuracy = rf.score(test_data,test_target)
    
    print(f"\nRegression Forest R^2 {accuracy:.3f}\n\n\n")
    
    return rf

def inp(df, exclude_columns=['est__', 't_all__', 't_gen__', 't_opt__', 't_lto__','relabel']):
    return df.drop(columns=list(set(exclude_columns) & set(df.columns)))

def out(df, target_column='t_all__'):
    return df[target_column]

def relable(df, model):
    df = df.copy()
    df['relabel'] = model.predict(inp(df))
    return df
    
def distilled(data, rf):
    # Split data into three four pieces:
    #   train to train the forest
    #   future to not be seen for either model
    #   and all_data to be relabeled by the forest, it includes some data the forest does not see
    train, rest = train_test_split(data,test_size=.3)
    future, test  = train_test_split(rest, test_size=0.5)
    all_data = pd.concat((train, test))
    
    rf.fit(inp(train),out(train))
    
    # Create a new column that contains just the RF predictions
    relabled = relable(all_data,rf)
    
    # Overfit a single tree to match the RF predictions
    dt = DecisionTreeRegressor(max_depth=None)
    dt.fit(inp(relabled), out(relabled, 'relabel'))
    
    # Now both models are tested on withheld data
    print("Performance on withheld data")
    print("\nRandom Forest R^2")
    print(rf.score(inp(future),out(future)))
    
    print("\nDecision Tree R^2")
    print(dt.score(inp(future),out(future)))
    #print(dt.get_depth())
    
    print("\nExisting Estimator R^2")
    print(r2_score(future['est__'],future['t_all__']))
    
    #plt.figure(figsize=(45,35))
    #sktree.plot_tree(dt,feature_names=relevant_features)
    
    rf, dt

#params_via_cv(data[0])
dbg_fitted_model = RandomForestRegressor(ccp_alpha = 0.2, min_samples_split = 2, max_features = 2, n_estimators = 100)
#params_via_cv(data[1])
opt_fitted_model = RandomForestRegressor(ccp_alpha = 0.3, min_samples_split = 2, max_features = 4, n_estimators = 100)




print("Debug Data")
#analyze(data[0],dbg_fitted_model)
distilled(data[0], dbg_fitted_model)

print("\n\n")
print("\n\nOpt Data")
#analyze(data[1], opt_fitted_model)
distilled(data[1], opt_fitted_model)
