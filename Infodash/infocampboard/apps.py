from django.apps import AppConfig


class InfocampboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'infocampboard'
    
    def ready(self):
        import infocampboard.dash_apps
        import infocampboard.infodash
