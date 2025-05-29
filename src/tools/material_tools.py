"""
Tool implementations for SimpleReActFramework.
"""
import os
import traceback
import pandas as pd
import numpy as np
from matminer.featurizers.composition import ElementProperty
from catboost import CatBoostClassifier
from pymatgen.core.composition import Composition
import PyPDF2
import re
from tqdm import tqdm
from openai import OpenAI

from src.utils.helpers import sanitize_path, safe_composition_conversion, ensure_directory_exists
from src.utils.logger import setup_logger
from config.settings import DEFAULT_BASE_URL, DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE, TRAINING_CONFIG

logger = setup_logger("material_analysis.tools.material_tools")

class MaterialTools:
    def __init__(self):
        self.data = None
        self.df_magpie = None
        self.X = None
        self.rule_match_materials = None
        self.client = None
        self.max_chunk_size = 4000
        self.overlap_size = 200
        
    def set_openai_client(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=DEFAULT_BASE_URL
        )

    def feature_engineering(self, data_path: str) -> str:
        data_path = data_path.strip().replace('```', '').strip()
        try:
            if not os.path.exists(data_path):
                return f"Data path not found: {data_path}"
            
            data_dir = os.path.dirname(data_path)
            file_name = os.path.basename(data_path)
            base_name = os.path.splitext(file_name)[0]
            
            train_df = pd.read_csv(data_path)
            if 'Substance' not in train_df.columns or 'label' not in train_df.columns:
                return "Data file must contain 'Substance' and 'label' columns"
            
            stats_df = self.create_element_stats_df(train_df)
            weighted_save_path = os.path.join(data_dir, f"{base_name}_weighted.csv")
            result = self.save_weighted_results(train_df, stats_df, weighted_save_path)
            
            return f"Processing completed:\n{result}"
            
        except Exception as e:
            return f"Processing failed: {str(e)}"

    def model_train(self, data_path: str) -> str:
        data_path = data_path.strip().replace('```', '').strip()
        try:
            from src.models.trainer import ModelTrainer
            from src.data.processor import DataProcessor
            
            if not data_path:
                return "data_path is required"
            
            data_path = data_path.strip().replace('```', '').strip()
            data_path = sanitize_path(data_path)
            model_save_path = os.path.join('model', 'model.cbm')
            
            if not os.path.exists(data_path):
                return f"Data file not found: {data_path}"
            
            data_processor = DataProcessor()
            
            X_train, X_test, y_train, y_test = data_processor.load_and_split_data(
                data_path,
                target_column='label',
                test_size=TRAINING_CONFIG['TEST_SIZE'],
                random_state=TRAINING_CONFIG['RANDOM_STATE']
            )
            
            trainer = ModelTrainer(random_state=TRAINING_CONFIG['RANDOM_STATE'])
            
            trainer.train(X_train, y_train)
            
            trainer.evaluate(X_test, y_test, log_metrics=False)
            
            ensure_directory_exists('model')
            trainer.save_model(model_save_path)
            
            return f"Model training completed successfully. Model saved to {model_save_path}"
            
        except Exception as e:
            return f"Model training failed: {str(e)}"

    def model_predict(self, data_path: str) -> str:
        try:
            data_path = data_path.strip().replace('```', '').strip()
            data_path = sanitize_path(data_path)
            model_path = os.path.join('model', 'model.cbm')
            
            if not os.path.exists(model_path):
                return f"Model file not found: {model_path}"
                
            if not os.path.exists(data_path):
                return f"Data file not found: {data_path}"
            
            data = pd.read_csv(data_path)
            if 'Substance' not in data.columns:
                return "Data file must contain 'Substance' column"
            
            weight_dict_path = os.path.join(os.path.dirname(data_path), 'weight_dict.json')
            if not os.path.exists(weight_dict_path):
                return f"Weight dictionary not found at {weight_dict_path}. Please process training data first."
            
            import json
            with open(weight_dict_path, 'r') as f:
                weight_dict = json.load(f)
            
            data['weighted_formula'] = data['Substance'].apply(
                lambda x: self.weight_formula(x, weight_dict)
            )
            
            data['composition'] = data['weighted_formula'].apply(safe_composition_conversion)
            ep_featurizer = ElementProperty.from_preset('magpie')
            df_magpie = ep_featurizer.featurize_dataframe(data, col_id='composition')
            X = df_magpie.drop(['Substance', 'composition', 'weighted_formula'], axis=1)
            
            loaded_model = CatBoostClassifier()
            loaded_model.load_model(model_path)
            y_pred = loaded_model.predict_proba(X)[:,1]
            
            data['pred'] = y_pred
            data = data.sort_values(by=['pred'], ascending=False)
            
            base_name = os.path.splitext(data_path)[0]
            result_path = f"{base_name}_predict.csv"
            data = data.drop(['composition', 'weighted_formula'], axis=1)  # 删除中间处理列
            data.to_csv(result_path, index=False)
            
            return f"Model prediction completed successfully. Results saved to {result_path}"
            
        except Exception as e:
            return f"Model prediction failed: {str(e)}"
        
    def criterion_match(self, data_path: str) -> str:
        data_path = data_path.strip().replace('```', '').strip()
        try:
            if not os.path.exists(data_path):
                return f"Data file not found: {data_path}"
            
            data = pd.read_csv(data_path)
            if 'Substance' not in data.columns:
                return "Data file must contain 'Substance' column"
            
            rare_earth_elements = ['Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 
                                 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
            
            data['composition'] = data['Substance'].apply(safe_composition_conversion)
            
            def contains_rare_earth(comp):
                if not isinstance(comp, Composition):
                    return False
                elements = comp.elements
                for element in elements:
                    if element.symbol in rare_earth_elements:
                        return True
                return False
            
            data['rule_match'] = data['composition'].apply(contains_rare_earth)
            rare_earth_data = data[data['rule_match'] == True]
            
            base_name = os.path.splitext(data_path)[0]
            result_path = f"{base_name}_rare_earth.csv"
            rare_earth_data = rare_earth_data.drop(['composition', 'rule_match'], axis=1)
            rare_earth_data.to_csv(result_path, index=False)
            
            return f"Rule matching completed successfully, found {len(rare_earth_data)} materials containing rare earth elements. Results saved to {result_path}"
            
        except Exception as e:
            return f"Rule matching failed: {str(e)}"

    def read_pdf(self, pdf_path: str) -> list:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.max_chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.overlap_size
        
        return chunks
    
    def extract_from_chunk(self, chunk: str) -> str:
        prompt = """
        Extract information from this section of the paper, focusing ONLY on materials that exhibit enzyme-like activities.
        Return the results directly in CSV format with the following columns:
        Material Name,Chemical Formula,Material Description,Modifications,Enzyme Activity Types,Enzyme Activity Details

        Important:
        1. ONLY include materials that have confirmed enzyme-like activities (such as peroxidase-like, oxidase-like, catalase-like, etc.).
        2. If a material does not show any enzyme-like activity, DO NOT include it in the results.

        Guidelines:
        1. Chemical Formula column:
        - Provide ONLY the PURE chemical formula using standard element symbols (e.g., Fe2O3, ZnO)
        - **DO NOT include any charges, oxidation states, brackets, dashes, or abbreviations**
        - **DO NOT use any organic group abbreviations like BDC, mim, etc.**
        - **DO NOT include parentheses ( ), brackets [ ], or hyphens -**
        - **Expand all coordination compounds or MOFs into their complete atomic formulas, containing ONLY element symbols and numbers**
        - For materials with multiple chemical formulas, separate them with semicolons (e.g., Fe3O4;Fe2O3)
        - Example (correct): Zn4O13C24H12
        - Example (incorrect): Zn4O(C8H4O4)3, Zn4O(BDC)3, ZIF-8

        2. Material Description column:
        - Describe the material's structure, composition, and key physical/chemical properties
        - Avoid using shorthand material names (e.g., ZIF-8), and instead describe the structure fully

        3. General guidelines:
        - DO NOT include any markdown formatting or code block markers
        - Return ONLY the CSV content, no additional text or formatting

        Example output format:
        Material Name,Chemical Formula,Material Description,Modifications,Enzyme Activity Types,Enzyme Activity Details
        Magnetite Composite,Fe3O4;Fe2O3,Fe3O4 nanoparticles encapsulated in a Zn-based porous coordination network with high surface area,,Peroxidase-like activity,Catalyzes oxidation of TMB in presence of H2O2
        Gold-Silver Core-Shell,Au;Ag,Gold core with silver shell structure providing synergistic catalytic performance,,Peroxidase-like activity,Catalyzes substrate oxidation in presence of H2O2

        If no materials with enzyme-like activities are found, return an empty string.

        Section text:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=DEFAULT_MODEL_NAME,
                temperature=DEFAULT_TEMPERATURE,
                messages=[
                    {"role": "system", "content": "You are an expert in materials science and enzymology with exceptional skills in extracting and analyzing information from scientific literature. You excel at identifying and categorizing materials with enzyme-like activities. Return results in CSV format only."},
                    {"role": "user", "content": prompt + chunk}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error extracting information: {str(e)}")
            return ""
    
    def process_paper(self, pdf_path: str):
        chunks = self.read_pdf(pdf_path)
        results = []
        
        for chunk in tqdm(chunks):   
            result = self.extract_from_chunk(chunk)
            if result:
                lines = result.split('\n')
                results.extend(lines[1:])
        
        return results

    def information_extract(self, input_dir: str) -> str:
        input_dir = input_dir.strip().replace('```', '').strip()
        
        try:
            if not os.path.exists(input_dir):
                return f"Directory not found: {input_dir}"
                
            if self.client is None:
                return "OpenAI client not initialized. Please set API key first."
            
            ensure_directory_exists(input_dir)
            pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                return f"No PDF files found in directory: {input_dir}"
            
            all_results = []
            all_results.append('Material Name,Chemical Formula,Material Description,Modifications,Enzyme Activity Types,Enzyme Activity Details,Literature Reference')
            
            for pdf_file in pdf_files:
                seen_materials = set()
                pdf_path = os.path.join(input_dir, pdf_file)
                results = self.process_paper(pdf_path)
                
                literature_ref = pdf_file[:-4]
                
                for result in results:
                    if isinstance(result, str):
                        row = result
                    else:
                        row = ','.join(result)
                    
                    material_name = row.split(',')[0].strip()
                    
                    if material_name not in seen_materials:
                        seen_materials.add(material_name)
                        all_results.append(f"{row},{literature_ref}")
            
            output_file = os.path.join(input_dir, "literature.csv")
            with open(output_file, 'w', encoding='utf-8') as f:
                for row in all_results:
                    f.write(row + '\n')
            
            return f"Successfully processed {len(pdf_files)} files. Results saved to {output_file}"
            
        except Exception as e:
            return f"Information extraction failed: {str(e)}"
        
    def create_element_stats_df(self, train_df):
        def extract_composition(formula):
            return Composition(formula).as_dict()

        element_stats = {}
        
        for formula in train_df['Substance'].unique():
            composition = extract_composition(formula)
            formula_label = train_df[train_df['Substance'] == formula]['label'].iloc[0]
            
            for element in composition:
                if element not in element_stats:
                    element_stats[element] = {'POD': 0, 'Non POD': 0, 'All': 0}
                
                element_stats[element]['All'] += 1
                if formula_label == 1:
                    element_stats[element]['POD'] += 1
                else:
                    element_stats[element]['Non POD'] += 1
        
        stats_df = pd.DataFrame([
            {
                'Element': element,
                'POD': f"{stats['POD']} ({stats['POD']/stats['All']*100:.1f})",
                'Non POD': f"{stats['Non POD']} ({stats['Non POD']/stats['All']*100:.1f})",
                'All': str(stats['All'])
            }
            for element, stats in element_stats.items()
        ])
        
        return stats_df

    def get_contrast_ratio(self, ratio, exp):
        if ratio < 1:
            return 1 / ratio
        if exp == 'none':
            return ratio
        elif exp == 'exp/2':
            return np.exp(ratio/2)
        elif exp == 'exp':
            return np.exp(ratio)
        else:
            raise ValueError("Invalid exp parameter")

    def get_contrast_ratio_dict(self, df, exp, contrast_epsilon):
        nonPOD_ratio = df['Non POD'].apply(lambda x: float(re.search(r'(\d+)\s*\(', x).group(1)) + contrast_epsilon)
        POD_ratio = df['POD'].apply(lambda x: float(re.search(r'(\d+)\s*\(', x).group(1)) + contrast_epsilon)
        ratio = (POD_ratio/ nonPOD_ratio).apply(self.get_contrast_ratio, exp=exp) / 3
        return dict(zip(df['Element'].tolist(), ratio))

    def compute_adjustment(self, df_t, alpha=0.1, beta=300):
        return 1 / (1 + np.exp(-alpha * (df_t - beta)))

    def get_idf_ratio_dict(self, df, total_count, idf_epsilon, alpha=0.03, beta=75):
        element_count = df['All'].astype(int)
        idf_effect = np.log((total_count + idf_epsilon) / (element_count + idf_epsilon))
        adjustment = self.compute_adjustment(element_count, alpha, beta)
        ratio = idf_effect * adjustment * 3
        return dict(zip(df['Element'], ratio))

    def get_ratio_dict(self, df, exp='exp', epsilon=10, alpha=0.03, beta=75):
        total_count = df['All'].astype(int).sum()
        
        contrast_ratio_dict = self.get_contrast_ratio_dict(df, exp, epsilon)
        idf_ratio_dict = self.get_idf_ratio_dict(df, total_count, idf_epsilon=epsilon, alpha=alpha, beta=beta)
        
        return {k: contrast_ratio_dict[k] * idf_ratio_dict[k] for k in contrast_ratio_dict if k in idf_ratio_dict}

    def weight_formula(self, formula, weight_dict):
        composition = Composition(formula).as_dict()
        weighted_parts = []
        
        for element, count in composition.items():
            if element in weight_dict:
                weighted_count = count * weight_dict[element]
                weighted_parts.append(f"{element}{weighted_count:.2f}")
            else:
                weighted_parts.append(f"{element}{count}")
        
        return ''.join(weighted_parts)

    def get_weighted_formulas(self, df, stats_df):
        weight_dict = self.get_ratio_dict(stats_df)
        
        result_df = df.copy()
        result_df['weighted_formula'] = result_df['Substance'].apply(
            lambda x: self.weight_formula(x, weight_dict)
        )
        
        return result_df

    def save_weighted_results(self, train_df, stats_df, save_path):
        try:
            weighted_df = self.get_weighted_formulas(train_df, stats_df)
            result_df = weighted_df[['weighted_formula', 'label']].copy()
            result_df.columns = ['Substance', 'label']
            result_df.to_csv(save_path, index=False)
            return f"Weighted results saved to {save_path}, total {len(result_df)} records"
        except Exception as e:
            return f"Failed to save weighted results: {str(e)}"
        
    def importance_analysis(self, data_path: str) -> pd.DataFrame:
        try:
            model_path = './model/model.cbm'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            data_path = data_path.strip().replace('```', '').strip()
            data_path = sanitize_path(data_path)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            import pandas as pd
            data = pd.read_csv(data_path)

            if 'Substance' in data.columns:
                from src.data.processor import DataProcessor
                processor = DataProcessor()
                data = processor.process_new_data(data_path)
                X = data
            else:
                X = data
            model = CatBoostClassifier()
            model.load_model(model_path)
            import shap
            explainer = shap.TreeExplainer(model)
            interaction_values = explainer.shap_interaction_values(X)
            n_blocks_per_dim = 132 // 6
            summed_blocks = interaction_values.reshape(-1, n_blocks_per_dim, 6, n_blocks_per_dim, 6).sum(axis=(-1, -3))
            summed_blocks = abs(summed_blocks).mean(axis=0)
            feature_names = []
            for i in range(0, 132, 6):
                feature = X.columns.tolist()[i]
                feature = feature.split()[-1]
                feature_names.append(feature)
            import numpy as np
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.diag(summed_blocks)
            })
            return importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        except Exception as e:
            logger.error(f"{str(e)}\n{traceback.format_exc()}")
            raise