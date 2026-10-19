"""Write a reviewable CloudFront/VPC-origin plan; never call AWS or publish anything."""

import argparse
import json
from pathlib import Path


def ref(name):
    return {"Ref": name}


def attribute(name, key):
    return {"Fn::GetAtt": [name, key]}


def template():
    """Create disabled-by-default infrastructure without credentials or account IDs."""
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "CARE-Vision private judge portal HTTPS entry; no vision or media resources",
        "Parameters": {
            "ExistingInstanceArn": {
                "Type": "String",
                "AllowedPattern": r"arn:aws:ec2:[a-z0-9-]+:[0-9]{12}:instance/i-[0-9a-f]+",
            },
            "ExistingVpcId": {"Type": "AWS::EC2::VPC::Id"},
            "ExistingPrivateDns": {
                "Type": "String",
                "AllowedPattern": r"ip-[0-9-]+\.[a-z0-9.-]+\.internal",
            },
            "ManagedCloudFrontSecurityGroup": {
                "Type": "String",
                "Default": "",
                "AllowedPattern": r"(?:|sg-[0-9a-f]+)",
                "Description": "Verified AWS-created CloudFront VPC origins SG in the same VPC",
            },
            "EnableEndpoint": {
                "Type": "String",
                "Default": "false",
                "AllowedValues": ["false", "true"],
            },
        },
        "Conditions": {
            "ManagedGroupKnown": {
                "Fn::Not": [{"Fn::Equals": [ref("ManagedCloudFrontSecurityGroup"), ""]}]
            },
            "EndpointEnabled": {
                "Fn::And": [
                    {"Fn::Equals": [ref("EnableEndpoint"), "true"]},
                    {"Condition": "ManagedGroupKnown"},
                ]
            },
        },
        "Resources": {
            "PortalOriginGroup": {
                "Type": "AWS::EC2::SecurityGroup",
                "Properties": {
                    "GroupDescription": "CARE-Vision judge portal: CloudFront VPC origin only",
                    "VpcId": ref("ExistingVpcId"),
                    "SecurityGroupIngress": [],
                    # Explicit harmless egress avoids introducing the EC2 allow-all default.
                    "SecurityGroupEgress": [{"IpProtocol": "-1", "CidrIp": "127.0.0.1/32"}],
                    "Tags": [{"Key": "Name", "Value": "CARE-Vision judge portal origin"}],
                },
            },
            "PortalOriginIngress": {
                "Type": "AWS::EC2::SecurityGroupIngress",
                "Condition": "ManagedGroupKnown",
                "Properties": {
                    "GroupId": attribute("PortalOriginGroup", "GroupId"),
                    "IpProtocol": "tcp",
                    "FromPort": 8780,
                    "ToPort": 8780,
                    "SourceSecurityGroupId": ref("ManagedCloudFrontSecurityGroup"),
                    "Description": "Only the verified AWS-managed CloudFront VPC origin group",
                },
            },
            "PortalVpcOrigin": {
                "Type": "AWS::CloudFront::VpcOrigin",
                "Properties": {
                    "VpcOriginEndpointConfig": {
                        "Arn": ref("ExistingInstanceArn"),
                        "Name": {"Fn::Sub": "${AWS::StackName}-judge"},
                        "HTTPPort": 8780,
                        "HTTPSPort": 443,
                        "OriginProtocolPolicy": "http-only",
                        "OriginSSLProtocols": ["TLSv1.2"],
                    }
                },
            },
            "NoCache": {
                "Type": "AWS::CloudFront::CachePolicy",
                "Properties": {
                    "CachePolicyConfig": {
                        "Name": {"Fn::Sub": "${AWS::StackName}-no-cache"},
                        "MinTTL": 0,
                        "DefaultTTL": 0,
                        "MaxTTL": 0,
                        "ParametersInCacheKeyAndForwardedToOrigin": {
                            "EnableAcceptEncodingGzip": False,
                            "EnableAcceptEncodingBrotli": False,
                            "HeadersConfig": {"HeaderBehavior": "none"},
                            "CookiesConfig": {"CookieBehavior": "none"},
                            "QueryStringsConfig": {"QueryStringBehavior": "none"},
                        },
                    }
                },
            },
            "ViewerRequest": {
                "Type": "AWS::CloudFront::OriginRequestPolicy",
                "Properties": {
                    "OriginRequestPolicyConfig": {
                        "Name": {"Fn::Sub": "${AWS::StackName}-viewer-request"},
                        "HeadersConfig": {"HeaderBehavior": "allViewer"},
                        "CookiesConfig": {"CookieBehavior": "all"},
                        "QueryStringsConfig": {"QueryStringBehavior": "none"},
                    }
                },
            },
            "PortalDistribution": {
                "Type": "AWS::CloudFront::Distribution",
                "Properties": {
                    "DistributionConfig": {
                        "Enabled": {"Fn::If": ["EndpointEnabled", True, False]},
                        "Comment": "CARE-Vision authenticated image-free judge portal",
                        "HttpVersion": "http2",
                        "IPV6Enabled": True,
                        "PriceClass": "PriceClass_100",
                        "ViewerCertificate": {"CloudFrontDefaultCertificate": True},
                        "Origins": [
                            {
                                "Id": "judge-portal",
                                "DomainName": ref("ExistingPrivateDns"),
                                "ConnectionAttempts": 2,
                                "ConnectionTimeout": 5,
                                "VpcOriginConfig": {
                                    "VpcOriginId": attribute("PortalVpcOrigin", "Id"),
                                    "OriginReadTimeout": 30,
                                    "OriginKeepaliveTimeout": 5,
                                },
                            }
                        ],
                        "DefaultCacheBehavior": {
                            "TargetOriginId": "judge-portal",
                            "ViewerProtocolPolicy": "https-only",
                            "AllowedMethods": [
                                "GET",
                                "HEAD",
                                "OPTIONS",
                                "PUT",
                                "PATCH",
                                "POST",
                                "DELETE",
                            ],
                            "CachedMethods": ["GET", "HEAD"],
                            "Compress": False,
                            "CachePolicyId": ref("NoCache"),
                            "OriginRequestPolicyId": ref("ViewerRequest"),
                        },
                        "CustomErrorResponses": [
                            {"ErrorCode": code, "ErrorCachingMinTTL": 0}
                            for code in (403, 404, 500, 502, 503, 504)
                        ],
                    }
                },
            },
        },
        "Outputs": {
            "HttpsUrl": {
                "Value": {
                    "Fn::Join": ["", ["https://", attribute("PortalDistribution", "DomainName")]]
                }
            },
            "DistributionId": {"Value": ref("PortalDistribution")},
            "VpcOriginId": {"Value": attribute("PortalVpcOrigin", "Id")},
            "PortalSecurityGroupId": {"Value": attribute("PortalOriginGroup", "GroupId")},
        },
    }


def write_plan(output):
    output = Path(output).absolute()
    repository = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repository):
        raise ValueError("keep deployment plans and identifiers outside the code repository")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    target = output / "cloudfront-vpc-origin.json"
    target.write_text(json.dumps(template(), indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        target = write_plan(args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Prepared disabled HTTPS plan: {target}; no AWS calls or network changes made.")


if __name__ == "__main__":
    main()
