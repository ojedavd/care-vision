import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_judge_https", Path(__file__).parents[1] / "tools/prepare_judge_https.py"
)
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


def test_plan_has_no_default_public_activation_or_account_identifiers():
    data = PLAN.template()
    assert data["Parameters"]["EnableEndpoint"]["Default"] == "false"
    for name in ("ExistingInstanceArn", "ExistingVpcId", "ExistingPrivateDns"):
        assert "Default" not in data["Parameters"][name]
    enabled = data["Resources"]["PortalDistribution"]["Properties"]["DistributionConfig"]["Enabled"]
    assert enabled == {"Fn::If": ["EndpointEnabled", True, False]}
    assert {"Condition": "ManagedGroupKnown"} in data["Conditions"]["EndpointEnabled"]["Fn::And"]
    assert data["Parameters"]["ManagedCloudFrontSecurityGroup"]["Default"] == ""


def test_security_group_only_accepts_managed_origin_and_adds_no_internet_egress():
    resources = PLAN.template()["Resources"]
    group = resources["PortalOriginGroup"]["Properties"]
    assert group["SecurityGroupIngress"] == []
    assert group["SecurityGroupEgress"] == [{"IpProtocol": "-1", "CidrIp": "127.0.0.1/32"}]
    ingress = resources["PortalOriginIngress"]
    assert ingress["Condition"] == "ManagedGroupKnown"
    rule = ingress["Properties"]
    assert rule["SourceSecurityGroupId"] == {"Ref": "ManagedCloudFrontSecurityGroup"}
    assert "CidrIp" not in rule and "CidrIpv6" not in rule
    assert rule["FromPort"] == rule["ToPort"] == 8780
    assert not any(
        resource["Type"]
        in {
            "AWS::EC2::Instance",
            "AWS::EC2::Subnet",
            "AWS::EC2::Route",
            "AWS::EC2::RouteTable",
            "AWS::S3::Bucket",
            "AWS::IAM::User",
            "AWS::IAM::AccessKey",
        }
        for resource in resources.values()
    )


def test_viewer_https_host_origin_cookie_csrf_and_post_are_preserved_without_cache():
    resources = PLAN.template()["Resources"]
    cache = resources["NoCache"]["Properties"]["CachePolicyConfig"]
    assert cache["MinTTL"] == cache["DefaultTTL"] == cache["MaxTTL"] == 0
    forward = resources["ViewerRequest"]["Properties"]["OriginRequestPolicyConfig"]
    assert forward["HeadersConfig"]["HeaderBehavior"] == "allViewer"
    assert forward["CookiesConfig"]["CookieBehavior"] == "all"
    distribution = resources["PortalDistribution"]["Properties"]["DistributionConfig"]
    behavior = distribution["DefaultCacheBehavior"]
    assert behavior["ViewerProtocolPolicy"] == "https-only"
    assert "POST" in behavior["AllowedMethods"]
    assert behavior["CachePolicyId"] == {"Ref": "NoCache"}
    assert behavior["OriginRequestPolicyId"] == {"Ref": "ViewerRequest"}
    assert all(e["ErrorCachingMinTTL"] == 0 for e in distribution["CustomErrorResponses"])
    assert "Logging" not in distribution
    assert "CustomOriginConfig" not in distribution["Origins"][0]
    assert distribution["Origins"][0]["VpcOriginConfig"]["VpcOriginId"] == {
        "Fn::GetAtt": ["PortalVpcOrigin", "Id"]
    }


def test_preparation_only_writes_new_external_files_without_network(tmp_path, monkeypatch):
    import socket
    import subprocess

    def reject(*args, **kwargs):
        raise AssertionError("preparation cannot call the network or AWS CLI")

    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(subprocess, "run", reject)
    path = PLAN.write_plan(tmp_path / "private-plan")
    assert json.loads(path.read_text()) == PLAN.template()
    initial = path.read_bytes()
    with pytest.raises(FileExistsError):
        PLAN.write_plan(path.parent)
    assert path.read_bytes() == initial
    with pytest.raises(ValueError):
        PLAN.write_plan(Path(__file__).parents[1] / "should-not-be-created")
