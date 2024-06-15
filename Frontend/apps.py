from django.apps import AppConfig


class FrontendConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Frontend'
    
    def ready(self):
        import Frontend.dashboard_main_view
