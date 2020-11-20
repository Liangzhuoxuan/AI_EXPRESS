from django.shortcuts import HttpResponse
import json
from .route_distribution import Router

def send_json(request):
    r = Router()
    j = r.run()
    print(j)
    return HttpResponse(json.dumps(j), content_type='application/json')