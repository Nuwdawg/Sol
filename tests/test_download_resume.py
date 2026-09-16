import base64
import hashlib
import pytest
import requests
from scripts.download_kernel import download_file


class Response:
    def __init__(self,data,start=0,fail=False):
        self.data,self.start,self.fail=data,start,fail
        self.status_code=206 if start else 200
        self.headers={'ETag':hashlib.md5(data).hexdigest(),'Content-Length':str(len(data)-start),
                      'x-goog-hash':'md5='+base64.b64encode(hashlib.md5(data).digest()).decode()}
        if start:self.headers['Content-Range']=f'bytes {start}-{len(data)-1}/{len(data)}'
    def raise_for_status(self):pass
    def close(self):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_content(self,n):
        if self.fail:
            yield self.data[:4]
            raise requests.exceptions.ChunkedEncodingError('interrupted')
        yield self.data[self.start:]


@pytest.mark.parametrize('kaggle_proxy',[False,True])
def test_resume_and_verified_skip(tmp_path,monkeypatch,kaggle_proxy):
    payload=b'abcdefghijk';calls=[]
    def get(url,**kwargs):
        headers=kwargs['headers'];calls.append(headers)
        start=int(headers.get('Range','bytes=0-')[6:-1])
        response=Response(payload,start,fail=len(calls)==1)
        if kaggle_proxy:
            response.headers.pop('ETag');response.headers.pop('x-goog-hash')
        return response
    monkeypatch.setattr(requests,'get',get)
    target=tmp_path/'weights.pt'
    url='https://example.invalid/kf/123/fake-access-token/weights.pt' if kaggle_proxy else 'unused'
    with pytest.raises(requests.exceptions.ChunkedEncodingError):download_file(url,target)
    assert not target.exists()
    report=download_file(url,target)
    assert calls[-1]['Range']=='bytes=4-'
    assert target.read_bytes()==payload
    again=download_file(url,target,report)
    assert again==report and len(calls)==4


def test_changed_remote_discards_partial(tmp_path,monkeypatch):
    payload=b'old-checkpoint';state=[True]
    def get(url,**kwargs):return Response(payload,fail=state[0])
    monkeypatch.setattr(requests,'get',get)
    target=tmp_path/'weights.pt'
    with pytest.raises(requests.exceptions.ChunkedEncodingError):download_file('unused',target)
    payload=b'new-checkpoint';state[0]=False
    download_file('unused',target)
    assert target.read_bytes()==payload
