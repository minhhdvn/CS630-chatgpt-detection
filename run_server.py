from model_loader import load_detection_scorer
import os
from torch.multiprocessing import *
import queue
import torch.cuda
import web
import json
import time

class Worker(Process):
    def __init__(self, job_queue, rersult_dict, device, id=0, timeout=5, batch_size=1):
        super(Worker, self).__init__()

        self.job_queue = job_queue
        self.result_dict = rersult_dict
        self.device = device
        self.id = id
        self.timeout = timeout
        self.batch_size = batch_size
        self.name = '{}/{}'.format('cuda:{}'.format(device) if device >= 0 else 'cpu', id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    def run(self):
        model = load_detection_scorer('logs/hc3.best-model.mdl')
        print(f'Worker {self.name} is ready')
        while True:
            try:
                data = []
                for _ in range(self.batch_size):
                    rid, ex = self.job_queue.get()
                    data.append([rid, ex])

                    print(f'Worker: {self.name} handles request', rid)

                result = self.handle(model, data)
                for rid in result:
                    self.result_dict[rid] = result[rid]

            except queue.Empty:
                pass

    def handle(self, model, request_data):
        questions, answers, rids = [], [], []
        for rid, ex in request_data:
            questions.append(ex['question'] if 'question' in ex else '')
            answers.append(ex['answer'] if 'answer' in ex else '')
            rids.append(rid)
            
        detection_scores = model.compute_detection_scores(questions, answers)
        predictions = []
        result = {}
        
        for rid, score in zip(rids, detection_scores):
            if score >= 0.5:
                pred = {'machine-generated': 'Yes', 'confidence': score}
            else:
                pred = {'machine-generated': 'No', 'confidence': 1 - score}
            result[rid] = pred

        return result


class MultiWorker(object):
    def __init__(self, job_queue, result_dict, num_gpus=8, instances_per_gpu=4):
        super(MultiWorker, self).__init__()
        assert num_gpus >= 0
        assert instances_per_gpu > 0

        self.job_queue = job_queue
        self.result_dict = result_dict
        self.num_gpus = num_gpus
        self.instances_per_gpu = instances_per_gpu

        self.workers = []
        if self.num_gpus == 0 or torch.cuda.device_count() == 0:
            print('Running on CPU...')
            for instance_id in range(self.instances_per_gpu):
                w = Worker(self.job_queue, self.result_dict, device=-1, id=instance_id)
                w.start()
                self.workers.append(w)
        else:
            avail_gpus = torch.cuda.device_count()
            print('{} GPUs are available'.format(avail_gpus))
            usable_gpus = min(self.num_gpus, avail_gpus)
            print('Using {} GPUs...'.format(usable_gpus))
            for gpu_id in range(usable_gpus):
                for instance_id in range(self.instances_per_gpu):
                    w = Worker(self.job_queue, self.result_dict, device=gpu_id, id=instance_id)
                    w.start()
                    self.workers.append(w)
        print('Initialized {} workers!'.format(len(self.workers)))

    def handle(self, request_data):
        timeout = 5
        interval = 0.01
        rid = time.time()
        self.job_queue.put((rid, request_data))
        print('Multiworker put: ', rid)
        i = 0
        while i < timeout:
            if rid in self.result_dict:
                result = self.result_dict.pop(rid)
                return result
            i += interval
            time.sleep(interval)
        return {'msg': 'Timeout'}

class GPTDetection:

    def POST(self):
        request_data = json.loads(web.data())
        result = workers.handle(request_data)
        return json.dumps(result)


if __name__ == '__main__':
    set_start_method('spawn')
    ctx = get_context('spawn')
    manager = ctx.Manager()
    job_queue = manager.Queue()
    result_dict = manager.dict()
    ngpus = torch.cuda.device_count()

    workers = MultiWorker(job_queue, result_dict, num_gpus=ngpus, instances_per_gpu=1)
    urls = (
        '/', 'GPTDetection',
    )
    app = web.application(urls, globals())
    web.httpserver.runsimple(app.wsgifunc(), ("127.0.0.1", 1234))
