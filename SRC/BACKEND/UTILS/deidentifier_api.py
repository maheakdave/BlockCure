import time
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines.token_classification import TokenClassificationPipeline
import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import tensorflow as tf
import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

try:
    model_checkpoint = "Davlan/bert-base-multilingual-cased-ner-hrl"
    logger.info("Loading model and tokenizer from %s", model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    logger.info("Model and tokenizer loaded successfully")

except Exception as e:
    logger.error("Error loading model or tokenizer: %s", e)
    raise e


class TokenClassificationChunkPipeline(TokenClassificationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, sentence, offset_mapping=None, **preprocess_params):
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,  # Return multiple chunks
            max_length=self.tokenizer.model_max_length,
            padding=True
        )
        #inputs.pop("overflow_to_sample_mapping", None)
        num_chunks = len(inputs["input_ids"])

        for i in range(num_chunks):
            if self.framework == "tf":
                model_inputs = {k: tf.expand_dims(v[i], 0) for k, v in inputs.items()}
            else:
                model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            if offset_mapping is not None:
                model_inputs["offset_mapping"] = offset_mapping
            model_inputs["sentence"] = sentence if i == 0 else None
            model_inputs["is_last"] = i == num_chunks - 1
            yield model_inputs

    def _forward(self, model_inputs):
        
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")

        overflow_to_sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        output = self.model(**model_inputs)
        logits = output["logits"] if isinstance(output, dict) else output[0]


        model_outputs = {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            "is_last": is_last,
            **model_inputs,
        }

        model_outputs["input_ids"] = torch.reshape(model_outputs["input_ids"], (1, -1))
        model_outputs["token_type_ids"] = torch.reshape(model_outputs["token_type_ids"], (1, -1))
        model_outputs["attention_mask"] = torch.reshape(model_outputs["attention_mask"], (1, -1))
        model_outputs["special_tokens_mask"] = torch.reshape(model_outputs["special_tokens_mask"], (1, -1))
        model_outputs["offset_mapping"] = torch.reshape(model_outputs["offset_mapping"], (1, -1, 2))

        return model_outputs

pipe = TokenClassificationChunkPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="simple")
# pipe = torch.compile(pipe)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    try:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        logger.error("Error loading spaCy model: %s", e)
        raise e

matcher1 = Matcher(nlp.vocab)
pattern = [{"TEXT":{"REGEX":"[a-zA-Z0-9_.]+@[a-zA-Z0-9_.]+"}}]
matcher1.add("EMAIL",[pattern])

nlp = English()
matcher = PhraseMatcher(nlp.vocab, attr="SHAPE")
matcher.add("ADMISSION_NUMBER", [nlp("ISP-0000-0000000000")])

matcher.add("DATE", [nlp("00-00-0000"),nlp("0-00-0000"),nlp("00-0-0000"),nlp("0-0-0000"),nlp("00-00-00"),nlp("0-00-00"),nlp("00-0-00"),nlp("0-0-00"),nlp("0000-00-00")])

matcher.add("PHONE_NUMBER", [nlp("(+91) 0000-000000"),nlp("(91) 0000-000000"),nlp("0000-000000")])

def anonymize(text:str)->str:
    """
    Masks the input text by replacing named entities with asterisks.
    """

    with torch.no_grad():
        ents = pipe(text)

    spans = [(e["start"], e["end"]) for e in ents]
    spans.sort(reverse=True)

    for start, end in spans:
        text = text[:start] + "*****" + text[end:]

    return text

def deidentify(text:str, nlp:spacy.language = nlp)-> str:
    """
    Takes in a text(passage, lines, etc.) and deidentifies it by removing PII
    """

    text = anonymize(text)
    #nlp = en_core_web_sm.load()
    
    doc= nlp(text)
    matches= matcher(doc)
    anonymized=""

    lst =[t.text for t in doc]
    for i in range(len(matches)):
        for j in range(matches[i][1],matches[i][2]):
            lst[j]="**"
    matches= matcher1(doc)
    for i in range(len(matches)):
        for j in range(matches[i][1],matches[i][2]):
            lst[j]="**"
    for ele in lst:
        anonymized+=ele+" "
    return anonymized

class patient(BaseModel):
    id:str
    diagnosis:str
    treatment: str

app = FastAPI()

@app.post("/api")
async def root(record:patient)-> JSONResponse:
    """
    API endpoint that deidentifies the treatment field in the patient record
    """
    
    logger.info("Received record with ID: %s", record.id)
    start_time = time.perf_counter()
    record.treatment = deidentify(record.treatment)
    logger.info("Processing time for record ID %s: %s", record.id, time.perf_counter() - start_time)
    logger.info("Deidentification completed for record ID: %s", record.id)

    jr_d = jsonable_encoder(record)
    return JSONResponse(content=jr_d,status_code=200,media_type="application/json")

if __name__ == "__main__":
    logger.info("Starting Uvicorn server on port 8010")
    uvicorn.run(app, port=8010, host="")