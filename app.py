'''from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load class labels
disease_classes = [
    "Diabetic Retinopathy", "Glaucoma", "Age-related Macular Degeneration",
    "Hypertensive Retinopathy", "Retinal Detachment", "Retinitis Pigmentosa",
    "Macular Hole", "Cytomegalovirus Retinitis", "Branch Retinal Vein Occlusion",
    "Central Serous Retinopathy", "Papilledema", "Myopic Degeneration",
    "Toxoplasmosis Scar", "Choroidal Neovascularization", "Vitreomacular Traction",
    "Optic Neuritis", "Uveitis", "Coloboma", "Retinal Vein Occlusion",
    "Coats' Disease"
]

# Load model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(disease_classes))
model.load_state_dict(torch.load("mured_retinal_disease_model.pt", map_location=torch.device('cpu')))
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(img_path)

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = disease_classes[predicted.item()]

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)'''

# first update
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Updated 20 disease labels
disease_classes = [
    "DR", "Normal", "ODC", "Other", "MH", "DN", "ARMD", "TSLN", "MYA", "BRVO",
    "ODP", "CNV", "RS", "ODE", "CRVO", "LS", "CSR", "HTR", "ASR", "CRS"
]

disease_details = { "DR": { "description": "Diabetic Retinopathy is a complication of diabetes that affects the eyes by damaging the blood vessels of the retina.", "treatment": "Treatment includes managing blood sugar levels, laser therapy, anti-VEGF injections, or vitrectomy depending on severity." }, "Normal": { "description": "No signs of disease or abnormalities; the retina and optic disc appear healthy.", "treatment": "No treatment necessary; routine eye check-ups are recommended to monitor eye health." }, "ODC": { "description": "Optic Disc Cupping is an increased cupping of the optic nerve head, commonly associated with glaucoma.", "treatment": "Treatment may involve eye drops, laser procedures, or surgery to lower intraocular pressure." }, "OTHER": { "description": "Represents miscellaneous eye conditions not categorized into the other specific labels.", "treatment": "Treatment varies depending on the specific underlying condition." }, "MH": { "description": "Macular Hole is a small break in the macula, leading to central vision loss.", "treatment": "Vitrectomy surgery with gas bubble injection is typically performed to repair the hole." }, "DN": { "description": "Diabetic Neuropathy refers to nerve damage caused by diabetes; it is less common in ophthalmic context.", "treatment": "Managing diabetes and possibly using medications to control pain or symptoms." }, "TSLN": { "description": "Tractional Retinal Detachment occurs when scar tissue pulls the retina away from the underlying tissue.", "treatment": "Vitrectomy surgery is often required to remove the scar tissue and reattach the retina." }, "ARMD": { "description": "Age-Related Macular Degeneration is a condition affecting older adults, causing central vision loss.", "treatment": "Anti-VEGF injections, photodynamic therapy, and lifestyle changes can help manage the condition." }, "MYA": { "description": "Myopia or nearsightedness is a common refractive error where distant objects appear blurred.", "treatment": "Corrective lenses, refractive surgery like LASIK, or orthokeratology can be used." }, "BRVO": { "description": "Branch Retinal Vein Occlusion is a blockage in one of the small veins in the retina.", "treatment": "Anti-VEGF injections, corticosteroids, or laser therapy can reduce swelling and improve vision." }, "ODP": { "description": "Optic Disc Pit is a congenital defect of the optic disc that may cause vision loss.", "treatment": "Observation or vitrectomy surgery if associated with macular detachment." }, "CNV": { "description": "Choroidal Neovascularization involves the growth of abnormal blood vessels in the choroid, often linked with wet ARMD.", "treatment": "Anti-VEGF injections are the primary treatment to stop the growth of abnormal vessels." }, "RS": { "description": "Retinal Scarring involves damage and fibrous tissue formation on the retina, often affecting vision.", "treatment": "Depends on the cause; may include laser surgery or vitrectomy." }, "ODE": { "description": "Optic Disc Edema is the swelling of the optic disc, often due to raised intracranial pressure.", "treatment": "Requires treating the underlying cause such as high intracranial pressure or inflammation." }, "CRVO": { "description": "Central Retinal Vein Occlusion is a blockage in the main vein of the retina leading to vision problems.", "treatment": "Anti-VEGF injections, corticosteroids, and laser therapy are commonly used." }, "LS": { "description": "Lattice Degeneration involves thinning and weakening of the peripheral retina.", "treatment": "Laser photocoagulation may be used to seal weak areas and prevent retinal detachment." }, "CSR": { "description": "Central Serous Retinopathy is a condition where fluid builds up under the retina, causing vision distortion.", "treatment": "Observation, laser treatment, or photodynamic therapy in chronic cases." }, "HTR": { "description": "Hypertensive Retinopathy refers to damage to the retinal blood vessels caused by high blood pressure.", "treatment": "Managing systemic hypertension is key; vision often improves with blood pressure control." }, "ASR": { "description": "Anterior Segment Retinopathy is a broad term for diseases affecting the front part of the eye including iris, cornea, and lens.", "treatment": "Treatment depends on the specific disease; may involve medication or surgery." }, "CRS": { "description": "Central Retinal Artery Stenosis involves narrowing of the main artery supplying the retina.", "treatment": "Emergency intervention is required; may include hyperbaric oxygen therapy or managing underlying causes." } }
       


# Load the model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(disease_classes))
model.load_state_dict(torch.load("mured_retinal_disease_model.pt", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(img_path)

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = disease_classes[predicted.item()]

    disease_info = disease_details.get(label, {"description": "Not available", "treatment": "Not available"})
    return render_template('result.html', prediction=label, description=disease_info["description"], treatment=disease_info["treatment"])


if __name__ == '__main__':
    app.run(debug=True)
