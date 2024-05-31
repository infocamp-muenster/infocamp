from django.apps import AppConfig


class InfocampboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'infocampboard'
    
    def ready(self):
        import infocampboard.infodash
        import infocampboard.real_time_view
        # import infocampboard.tweet_processing
