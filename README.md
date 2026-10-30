# PlantCare AI

A Flask-based plant disease screening website that compares user-described
symptoms with a plant pathology dataset.

## Features

- Select a crop and describe visible symptoms.
- Choose from five text classification or similarity methods.
- View up to three candidate disease matches.
- Review causal organisms, typical symptoms, and diagnostic notes.
- View general disease-management suggestions.
- Responsive interface for desktop and mobile screens.

## Dataset

The application expects this CSV file in the project root:

`plant_pathology_dataset (1).csv`

Required columns:

- Crop
- Disease
- Causal_organism
- Typical_symptoms
- Pathogen_group
- Spread_and_favourable_conditions
- Diagnostic_notes

## Setup

Create and activate a virtual environment.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py