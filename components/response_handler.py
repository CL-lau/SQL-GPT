from streamlit import logger

app_logger = logger.get_logger(__name__)

def get_response(query: str, model) -> str:
    app_logger.info(f'\033[36mUser Query: {query}\033[0m')
    try:
        if model is not None and query:
            response = model.run(query)
            app_logger.info(f'\033[36mLLM Response: {response}\033[0m')
            return response
        return (
            'Your model still not created.\n'
            '1. If you are using gpt4free model, '
            'try to re-select a provider. '
            '(Click the "Show Available Providers" button in sidebar)\n'
            '2. If you are using openai model, '
            'try to re-pass openai api key.\n'
            '3. Or you did not pass the file successfully.\n'
            '4. Try to Refresh the page (F5).'
        )
    except Exception as e:
        app_logger.info(f'{__file__}: {e}')
        return (
            'Something wrong in docGPT...\n'
            '1. If you are using gpt4free model, '
            'try to select the different provider. '
            '(Click the "Show Available Providers" button in sidebar)\n'
            '2. If you are using openai model, '
            'check your usage for openai api key.\n'
            '3. Try to Refresh the page (F5).'
        )
