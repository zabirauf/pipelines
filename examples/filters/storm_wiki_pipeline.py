"""
title: Storm Wiki Pipeline
author: zabirauf
date: 2024-07-22
version: 1.0
license: MIT
description: Research about a topic and create a wikipedia like content containing summary and information with varied perspective.
requirements: knowledge_storm
environment_variables: OPENAI_API_KEY, YOU_API_KEY, REGULAR_MODEL_NAME, SMART_MODEL_NAME
"""

import tempfile
import os

from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint

from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import YouRM
from pydantic import BaseModel, Field

class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        OPENAI_API_KEY: str = ""
        YOU_API_KEY: str = ""
        REGULAR_MODEL_NAME: str = "gpt-4o-mini"
        SMART_MODEL_NAME: str = "gpt-4o"
        pass


    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline
            
        def research_topic(
            self,
            topic: str,
        ) -> str:
            """
            Research about a topic and create a wikipedia like content containing summary and information with varied perspective.

            :param topic: The topic user want to research about
            :return: The wikipedia like content containing summary and more detailed information about the topic
            """
            if self.pipeline.valves.OPENAI_API_KEY == "":
                return "OpenAPI Key not set, ask the user to set it up."
            elif self.pipeline.valves.YOU_API_KEY == "":
                return "You.com API Key not set, ask the user to set it up."
            else:

                lm_configs = STORMWikiLMConfigs()

                openai_kwargs = {
                    'api_key': self.pipeline.valves.OPENAI_API_KEY,
                    'temperature': 1.0,
                    'top_p': 0.9,
                }
                regular_model_name = self.pipeline.valves.REGULAR_MODEL_NAME
                smart_model_name = self.pipeline.valves.SMART_MODEL_NAME
                conv_simulator_lm = OpenAIModel(model=regular_model_name, max_tokens=500, **openai_kwargs)
                question_asker_lm = OpenAIModel(model=regular_model_name, max_tokens=500, **openai_kwargs)
                outline_gen_lm = OpenAIModel(model=smart_model_name, max_tokens=400, **openai_kwargs)
                article_gen_lm = OpenAIModel(model=smart_model_name, max_tokens=700, **openai_kwargs)
                article_polish_lm = OpenAIModel(model=smart_model_name, max_tokens=4000, **openai_kwargs)

                lm_configs.set_conv_simulator_lm(conv_simulator_lm)
                lm_configs.set_question_asker_lm(question_asker_lm)
                lm_configs.set_outline_gen_lm(outline_gen_lm)
                lm_configs.set_article_gen_lm(article_gen_lm)
                lm_configs.set_article_polish_lm(article_polish_lm)

                # Create a temporary directory for output
                self.temp_output_dir = tempfile.mkdtemp()
                print(f"Created temporary output directory: {self.temp_output_dir}")

                # Update engine_args with the temporary directory
                engine_args = STORMWikiRunnerArguments(
                    output_dir=self.temp_output_dir,
                    max_conv_turn=3,
                    max_perspective=3,
                    search_top_k=3,
                    max_thread_num=3,
                )

                research_module = YouRM(ydc_api_key=self.pipeline.valves.YOU_API_KEY, k=5)
                runner = STORMWikiRunner(engine_args, lm_configs, research_module)
                print("Starting runner.run() with topic:", topic)
                runner.run(
                    topic=topic,
                    do_research=True,
                    do_generate_outline=True,
                    do_generate_article=True,
                    do_polish_article=True,
                )
                print("runner.run() completed")

                print("Starting runner.post_run()")
                runner.post_run()
                print("runner.post_run() completed")

                print("Starting runner.summary()")
                runner.summary()
                print("runner.summary() completed")

                # Read the polished article from the output directory
                article_path = os.path.join(runner.article_output_dir, 'storm_gen_article_polished.txt')
                print(f"Attempting to read polished article from: {article_path}")
                print(f"Checking if article exists at path: {article_path}")
                if os.path.exists(article_path):
                    print("Article found. Reading content...")
                    with open(article_path, 'r', encoding='utf-8') as f:
                        polished_article = f.read()
                    print(f"Article content read. Length: {len(polished_article)} characters")
                    return polished_article
                else:
                    print("Error: Polished article not found at the specified path.")
                    return "Error: Polished article not found."

    def __init__(self):
        super().__init__()

        self.name = "Storm Wiki Pipeline"
        self.valves = self.Valves(
            **{
                **self.valves.model_dump(),
                "pipelines": ["*"],  # Connect to all pipelines
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "YOU_API_KEY": os.getenv("YOU_API_KEY", ""),
                "REGULAR_MODEL_NAME": os.getenv("REGULAR_MODEL_NAME", "gpt-4o-mini"),
                "SMART_MODEL_NAME": os.getenv("SMART_MODEL_NAME", "gpt-4o"),
            },
        )
        self.tools = self.Tools(self)