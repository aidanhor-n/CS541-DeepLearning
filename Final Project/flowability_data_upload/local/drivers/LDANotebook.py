from flowability_data_upload.local.generate_datasets.loaddata import load_data
import flowability_data_upload.local.preprocess as preprocess
from flowability_data_upload.local.model.LDA import LDA
from flowability_data_upload.local.analytics.visualize_lda import VisualizeLDA

# Load the Data (load)
data = load_data()

# Call Preprocessing Steps (preprocess)

print("starting preprocessing")
data = preprocess.clean(data)
data = preprocess.balance_particles(data, seed=1)
data = preprocess.remove_identifying_columns(data)
y = preprocess.multiclass_bin(data.iloc[:, -1:], [15, 30])
print("Y", set(y))
x = preprocess.yeo_transform(data.iloc[:, :-1])
x, y = preprocess.get_numpy_data(x, y)
print("finish preprocessing")

# - Now you currently have a cleaned, yeo transformed, balanced pandas dataframe, labeled binary flowability

trained_models = list()

# Train with shrinkage (auto)
lda_model_auto = LDA(x, y, name="Auto")
trained_models.append(lda_model_auto)

# Train with shrinkage (0)
lda_model_no_shrink = LDA(x, y, shrinkage=0, name="No Shrinkage")
trained_models.append(lda_model_no_shrink)

# Train with full shrinkage (1)
lda_model_full_shrinkage = LDA(x, y, shrinkage=1, name="Full Shrinkage")
trained_models.append(lda_model_full_shrinkage)

# Validate Model/Visualize Model (analytics)

for lda_model in trained_models:
    results = VisualizeLDA(lda_model, x, y)
