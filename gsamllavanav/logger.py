from PIL import Image
import wandb

from gsamllavanav.parser import ExperimentArgs


_active = False
_silent = True


def init(args: ExperimentArgs):
    global _active
    global _silent
    
    _active = args.log
    _silent = args.silent
    
    if _active:
        if args.resume_log_id:
            wandb.init(entity='water-cookie', project='citynav', id=args.resume_log_id, resume="must")
        else:
            wandb.init(project='citynav', config=args.to_dict())


def define_metric(name: str, step_metric: str = None, summary: str = None):
    if _active:
        wandb.define_metric(name, step_metric, summary=summary)


def log(data, step=None, commit=None):
    if _active:
        wandb.log(data, step=step, commit=commit)
    if not _silent:
        print(data)


def log_images(name: str, images: list[Image.Image], captions: list[str], max_n=10, step=None, commit=None):
    if _active:
        images = [wandb.Image(image, caption=caption) for image, caption in zip(images[:max_n], captions[:max_n])]
        wandb.log({name: images}, step=step, commit=commit)


def finish():
    if _active:
        wandb.finish()
