terraform {
  required_version = ">= 1.5.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
  }
}

provider "kubernetes" {
  config_path    = var.kubeconfig
  config_context = var.kube_context
}

resource "kubernetes_namespace" "bitnet" {
  metadata {
    name = var.namespace
  }
}

resource "kubernetes_secret" "model" {
  metadata {
    name      = "bitnet-model"
    namespace = kubernetes_namespace.bitnet.metadata[0].name
  }

  data = {
    MODEL_URL = var.model_url
  }
}

resource "kubernetes_config_map" "runtime" {
  metadata {
    name      = "bitnet-runtime-config"
    namespace = kubernetes_namespace.bitnet.metadata[0].name
  }

  data = {
    INFERENCE_ARGS = var.inference_args
  }
}

resource "kubernetes_deployment" "bitnet" {
  metadata {
    name      = "bitnet-runtime"
    namespace = kubernetes_namespace.bitnet.metadata[0].name
    labels = {
      app = "bitnet-runtime"
    }
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = {
        app = "bitnet-runtime"
      }
    }

    template {
      metadata {
        labels = {
          app = "bitnet-runtime"
        }
      }

      spec {
        container {
          name              = "bitnet"
          image             = var.image
          image_pull_policy = var.image_pull_policy

          env {
            name = "MODEL_URL"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.model.metadata[0].name
                key  = "MODEL_URL"
              }
            }
          }

          env {
            name = "INFERENCE_ARGS"
            value_from {
              config_map_key_ref {
                name = kubernetes_config_map.runtime.metadata[0].name
                key  = "INFERENCE_ARGS"
              }
            }
          }

          args = [
            "run_inference_server.py",
            "--host",
            "0.0.0.0",
            "--model",
            "$(MODEL_URL)",
          ]

          port {
            name           = "http"
            container_port = var.container_port
          }

          resources {
            limits = var.resource_limits
            requests = var.resource_requests
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 15
            period_seconds        = 30
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 5
            period_seconds        = 10
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "bitnet" {
  metadata {
    name      = "bitnet-runtime"
    namespace = kubernetes_namespace.bitnet.metadata[0].name
    labels = {
      app = "bitnet-runtime"
    }
  }

  spec {
    selector = {
      app = "bitnet-runtime"
    }

    port {
      name        = "http"
      port        = var.service_port
      target_port = "http"
    }

    type = var.service_type
  }
}
