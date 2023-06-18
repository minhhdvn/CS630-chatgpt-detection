import requests
import json, time, multiprocessing


def test_request(poolname):
    start = time.time()
    question = 'write a short essay about covid 19'
    answer = '''The COVID-19 pandemic has been a major challenge for the world, and it has had a profound impact on our lives. As we have learned over the past few months, the virus can spread quickly and easily from person to person, and it can have serious health consequences. We have all had to make many changes in our daily lives to try to slow the spread of the disease, from social distancing to wearing masks in public. We have seen the impact of the pandemic on our economy, with businesses closing and jobs lost. We have also seen the importance of basic human needs, such as access to food and shelter, during this time of crisis.

The COVID-19 pandemic has reminded us of the importance of community and the need to work together to overcome challenges. We have seen volunteers stepping up to help those in need, and people coming together to support each other in difficult times. We have also seen the importance of science and evidence-based decision-making in response to the pandemic. As we move forward, it is important to continue to follow public health guidelines and to support efforts to develop a vaccine and treatments for the disease.

Overall, the COVID-19 pandemic has been a difficult time for all of us, but it has also brought out the best in humanity. We have seen the power of collaboration and compassion, and we have learned the importance of taking care of one another. As we move forward from this crisis, it is important to remember these lessons and to work together to build a better world for ourselves and for future generations.'''

    data = {'question': question, 'answer': answer}
    response = requests.post('http://127.0.0.1:1234', json=data)
    if response.status_code == 200:
        pred = response.json()
    else:
        pred = None
    print(pred)


def test_multiprocessing():
    num_p = 50
    num_r = 500


    total = 0
    for _ in range(num_r):
        start = time.time()
        p = multiprocessing.Process(target=test_request, args=('bob',))
        p.start()
        end = time.time()
        total += end - start

        time.sleep(1/num_p)
    print('Average response time (ms):', total/num_r * 1000)


if __name__ == '__main__':
    test_multiprocessing()
