variable "location" {
  type        = string
  description = "Location of the resources"
  default     = "us-central1"
}

variable "instance-type" {
  type        = string
  description = "Instance type to deploy"
  default     = "n1-standard-4"
}

variable "project" {
  type        = string
  description = "Project"
  default     = "cerulean-338116"
}
