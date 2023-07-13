from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/list/channels")
    assert response.status_code == 200
    # assert response.json() == {"msg": "Hello World"}

def test_read_main_2():
    response = client.get("/list_channeadawaadaws")
    assert response.status_code == 200
