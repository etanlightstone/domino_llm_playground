import os
import yaml
import glob

class TemplateManager:
    """Manages prompt templates for the LLM playground."""
    
    def __init__(self, config_dir='config'):
        """Initialize the template manager with a config directory."""
        self.config_dir = config_dir
        self.template_dir = os.path.join(config_dir, 'templates')
        
        # Create template directory if it doesn't exist
        os.makedirs(self.template_dir, exist_ok=True)
    
    def load_template(self, template_path):
        """Load a prompt template from a YAML file."""
        try:
            with open(template_path, 'r') as f:
                template = yaml.safe_load(f)
            return template
        except Exception as e:
            print(f"Error loading template {template_path}: {e}")
            return None

    def save_template(self, template_data, template_name):
        """Save a prompt template to a YAML file."""
        if not template_name.endswith('.yaml'):
            template_name += '.yaml'
        
        template_path = os.path.join(self.template_dir, template_name)
        try:
            with open(template_path, 'w') as f:
                yaml.dump(template_data, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error saving template: {e}")
            return False

    def delete_template(self, template_file):
        """Delete a template file."""
        try:
            template_path = os.path.join(self.template_dir, template_file)
            os.remove(template_path)
            return True
        except Exception as e:
            print(f"Failed to delete template: {e}")
            return False

    def list_templates(self):
        """List all available templates in the templates directory."""
        template_files = glob.glob(f'{self.template_dir}/*.yaml')
        templates = []
        
        for template_file in template_files:
            template = self.load_template(template_file)
            if template:
                templates.append({
                    'file': os.path.basename(template_file),
                    'name': template.get('name', os.path.basename(template_file)),
                    'description': template.get('description', ''),
                    'system_template': template.get('system_template', ''),
                    'human_template': template.get('human_template', '')
                })
        
        return templates
    
    def get_template_by_name(self, template_name):
        """Get a template by its name."""
        templates = self.list_templates()
        for template in templates:
            if template['name'] == template_name:
                return template
        return None
    
    def get_template_by_file(self, template_file):
        """Get a template by its filename."""
        template_path = os.path.join(self.template_dir, template_file)
        return self.load_template(template_path)
    
    def get_first_template(self):
        """Get the first available template, or create a default one if none exists."""
        templates = self.list_templates()
        
        if templates:
            return templates[0]
        
        # If no templates exist, create a default one and return it
        default_template = {
            'name': 'Default',
            'description': 'Default RAG assistant',
            'system_template': 'You are a helpful AI assistant answering questions based on the context.\n\nContext: {context}',
            'human_template': 'Question: {question}'
        }
        
        self.save_template(default_template, 'default.yaml')
        
        return {
            'file': 'default.yaml',
            'name': default_template['name'],
            'description': default_template['description'],
            'system_template': default_template['system_template'],
            'human_template': default_template['human_template']
        } 