import os

class DBConfig:
    """Database configuration for different environments."""
    
    ENV_CONFIGS = {
        'LOCAL': {
            'user': 'root',
            'password': 'netweb12',
            'host': 'localhost',
            'port': 3306,
            'database': 'rcm_service'
        },
        'DEV': {
            'user': 'root',  # Placeholder for DEV
            'password': 'netweb12',
            'host': 'dev-db-host',
            'port': 3306,
            'database': 'rcm_service'
        },
        'UAT': {
            'user': 'root',  # Placeholder for UAT
            'password': 'netweb12',
            'host': 'uat-db-host',
            'port': 3306,
            'database': 'rcm_service'
        },
        'PROD': {
            'user': 'root',  # Placeholder for PROD
            'password': 'netweb12',
            'host': 'prod-db-host',
            'port': 3306,
            'database': 'rcm_service'
        }
    }

    @staticmethod
    def get_config():
        """Get database configuration based on APP_ENV environment variable."""
        env = os.getenv('APP_ENV', 'LOCAL').upper()
        config = DBConfig.ENV_CONFIGS.get(env, DBConfig.ENV_CONFIGS['LOCAL'])
        
        # Allow environment variable overrides for secrets
        config['user'] = os.getenv('DB_USER', config['user'])
        config['password'] = os.getenv('DB_PASSWORD', config['password'])
        config['host'] = os.getenv('DB_HOST', config['host'])
        config['port'] = int(os.getenv('DB_PORT', config['port']))
        config['database'] = os.getenv('DB_NAME', config['database'])
        
        return config
