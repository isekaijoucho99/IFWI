import importlib.util
from pathlib import Path
import json
import hashlib
import pytest

ROOT=Path(__file__).resolve().parents[2]


def publisher():
    spec=importlib.util.spec_from_file_location('publisher',ROOT/'tools'/'publish_five_ideas.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


@pytest.mark.parametrize('url',['https://github.com/isekaijoucho99/IFWI.git','git@github.com:isekaijoucho99/IFWI.git','ssh://git@github.com/isekaijoucho99/IFWI.git'])
def test_publish_target_recognizes_only_user_repo(url):
    assert publisher().allowed_origin(url)


def test_publish_rejects_wrong_repo():
    m=publisher()
    for url in ['https://github.com/other/IFWI.git','https://github.com.attacker.invalid/isekaijoucho99/IFWI','/tmp/repo']:
        assert not m.allowed_origin(url)


def test_manifest_cannot_escape_or_overwrite(tmp_path):
    m=publisher(); package=tmp_path/'package'; worktree=tmp_path/'work'; package.mkdir(); worktree.mkdir()
    manifest=package/'DELIVERY_MANIFEST.json'
    manifest.write_text(json.dumps({'files':[{'path':'../bad.py','sha256':'a'}]}))
    with pytest.raises(ValueError,match='path'): m.validate_delivery(package,worktree)
    (package/'ok.py').write_text('x=1\n'); sha=hashlib.sha256((package/'ok.py').read_bytes()).hexdigest()
    manifest.write_text(json.dumps({'files':[{'path':'ok.py','sha256':sha}]}))
    assert m.validate_delivery(package,worktree)==['ok.py']
    (worktree/'ok.py').write_text('existing')
    with pytest.raises(FileExistsError): m.validate_delivery(package,worktree)
    assert (worktree/'ok.py').read_text()=='existing'


def test_publish_rejects_existing_dangling_symlink(tmp_path):
    m=publisher(); package=tmp_path/'pkg'; worktree=tmp_path/'tree'; package.mkdir(); worktree.mkdir()
    (package/'a.py').write_text('x=1\n')
    sha=hashlib.sha256((package/'a.py').read_bytes()).hexdigest()
    (package/'DELIVERY_MANIFEST.json').write_text(json.dumps({'files':[{'path':'a.py','sha256':sha}]}))
    try: (worktree/'a.py').symlink_to(worktree/'not_yet_existing.py')
    except OSError: pytest.skip('OS does not permit creating a symlink')
    with pytest.raises(FileExistsError): m.validate_delivery(package,worktree)
