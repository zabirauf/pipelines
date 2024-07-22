from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import subprocess
import tempfile
import os

from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.rm import YouRM
from knowledge_storm.utils import load_api_key

class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        OPENAI_API_KEY: str
        YOU_API_KEY: str
        REGULAR_MODEL_NAME: str = "gpt-4o-mini"
        SMART_MODEL_NAME: str = "gpt-4o"


    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "python_code_pipeline"
        self.name = "Storm Wiki Generator"
        self.lm_configs = STORMWikiLMConfigs()
        self.openai_kwargs = {
            'api_key': os.getenv("OPENAI_API_KEY"),
            'temperature': 1.0,
            'top_p': 0.9,
        }
        regular_model_name = os.getenv('REGULAR_MODEL_NAME', "gpt-4o-mini")
        smart_model_name = os.getenv('SMART_MODEL_NAME', "gpt-4o")
        conv_simulator_lm = ModelClass(model=regular_model_name, max_tokens=500, **openai_kwargs)
        question_asker_lm = ModelClass(model=regular_model_name, max_tokens=500, **openai_kwargs)
        outline_gen_lm = ModelClass(model=smart_model_name, max_tokens=400, **openai_kwargs)
        article_gen_lm = ModelClass(model=smart_model_name, max_tokens=700, **openai_kwargs)
        article_polish_lm = ModelClass(model=smart_model_name, max_tokens=4000, **openai_kwargs)

        self.lm_configs.set_conv_simulator_lm(conv_simulator_lm)
        self.lm_configs.set_question_asker_lm(question_asker_lm)
        self.lm_configs.set_outline_gen_lm(outline_gen_lm)
        self.lm_configs.set_article_gen_lm(article_gen_lm)
        self.lm_configs.set_article_polish_lm(article_polish_lm)

        # Create a temporary directory for output
        self.temp_output_dir = tempfile.mkdtemp()
        print(f"Created temporary output directory: {self.temp_output_dir}")

        # Update engine_args with the temporary directory
        self.engine_args = STORMWikiRunnerArguments(
            output_dir=self.temp_output_dir,
            max_conv_turn=3,
            max_perspective=3,
            search_top_k=3,
            max_thread_num=3,
        )

        self.research_module = YouRM(ydc_api_key=os.getenv('YOU_API_KEY'), k=5)


    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

        # This function is called when the server is stopped.
        # Delete the temporary output directory and its contents
        if hasattr(self, 'temp_output_dir') and os.path.exists(self.temp_output_dir):
            import shutil
            shutil.rmtree(self.temp_output_dir)
            print(f"Deleted temporary output directory: {self.temp_output_dir}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        if body.get("title", False):
            print("Title Generation")
            return "Storm Wiki Generation"

        runner = STORMWikiRunner(self.engine_args, self.lm_configs, self.research_module)
        runner.run(
            topic=user_message,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
        )
        runner.post_run()
        runner.summary()

        # Read the polished article from the output directory
        article_path = os.path.join(runner.article_output_dir, 'storm_gen_article_polished.txt')
        if os.path.exists(article_path):
            with open(article_path, 'r', encoding='utf-8') as f:
                polished_article = f.read()
            return polished_article
        else:
            return "Error: Polished article not found."