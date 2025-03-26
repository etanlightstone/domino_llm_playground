import os
import yaml

class SettingsManager:
    """Manages application settings for the LLM Playground."""
    
    def __init__(self, settings_file='config/llm_playground.yaml'):
        """Initialize the settings manager with a settings file path."""
        self.settings_file = settings_file
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from the YAML file."""
        default_settings = {
            'embeddings_model': 'multi-qa-mpnet-base-cos-v1',
            'selected_llm': 'llama2',
            'selected_template': 'Default',
            'selected_collections': [],
            'num_articles': 5,
            'ollama_api_url': 'http://localhost:11434'
        }
        
        if not os.path.exists(self.settings_file):
            return default_settings
        
        try:
            with open(self.settings_file, 'r') as f:
                settings = yaml.safe_load(f)
            
            # Ensure all expected settings are present
            for key, value in default_settings.items():
                if key not in settings:
                    settings[key] = value
                
            # Ensure selected_collections is a list
            if settings['selected_collections'] is None:
                settings['selected_collections'] = []
                    
            return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            return default_settings
    
    def save_settings(self, settings=None):
        """Save settings to the YAML file."""
        if settings:
            self.settings = settings
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        
        try:
            with open(self.settings_file, 'w') as f:
                yaml.dump(self.settings, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def get_setting(self, key, default=None):
        """Get a specific setting value."""
        return self.settings.get(key, default)
    
    def update_setting(self, key, value):
        """Update a specific setting value and save to file."""
        self.settings[key] = value
        return self.save_settings()
    
    def update_multiple_settings(self, settings_dict):
        """Update multiple settings at once and save to file."""
        self.settings.update(settings_dict)
        return self.save_settings() 