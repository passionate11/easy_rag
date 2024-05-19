from sentence_transformers import SentenceTransformer, models
from transformers.trainer import *


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True):
    print("-------------11---------------")
    word_embedding_model = models.Transformer(ckpt_dir)
    print("-------------12---------------")
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu', trust_remote_code=True)
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu', trust_remote_code=True)
    print("-------------13---------------")
    model.save(ckpt_dir)


class BiTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
        print("-------------8---------------")
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        print("-------------9---------------")
        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            print("-------------10---------------")
            save_ckpt_for_sentence_transformers(output_dir,
                                                pooling_mode=self.args.sentence_pooling_method,
                                                normlized=self.args.normlized)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # print(f'inputs herer is :{inputs}')
        # print(len(inputs))
        # for k,v in inputs.items():
        #     print(k)
        #     print(v)
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
