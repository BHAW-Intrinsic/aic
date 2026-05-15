FROM docker.io/library/ros:kilted-ros-core AS build

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install pixi
RUN apt update && apt install -y ca-certificates curl git && curl -fsSL https://pixi.sh/install.sh | env PIXI_VERSION=0.67.0 bash
ENV PATH="/root/.pixi/bin:$PATH"

# Add base dependencies
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

# Add other dependencies
ENV AIC_RSLRL_SC_POLICY_ARTIFACT=/ws_aic/src/aic/aic_model/artifacts/step6_sc_policy.pt
ENV AIC_RSLRL_SFP_POLICY_ARTIFACT=/ws_aic/src/aic/aic_model/artifacts/step11_sfp_gazebo_tight_a23f1da_model_100_policy.pt
ENV AIC_RSLRL_TASK_KIND=auto
ENV AIC_RSLRL_SFP_INCLUDE_MOUNT_METADATA=false
ENV AIC_RSLRL_REQUIRE_RESNET18=false
ENV AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_INSERT=true
ENV AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_TRACE_REPLAY=true
ENV AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_PATH=/ws_aic/src/aic/aic_model/artifacts/public_scripted_traces.json
ENV AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_TARGETS=nic_card_mount_0,nic_card_mount_1
ENV AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_DT=0.05
ENV AIC_RSLRL_PUBLIC_SCRIPTED_DWELL_SEC=0.0
ENV AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_FINAL_JOINTS=true
ENV AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_JOINT_TARGETS=sc_port_2
ENV AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_JOINT_STEPS=80
ENV AIC_RSLRL_PUBLIC_SCRIPTED_FINAL_JOINT_DT=0.05
ENV AIC_RSLRL_ENABLE_PUBLIC_SCRIPTED_FINAL_POSE_HOLD=false
ENV AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_DELTA_NIC_CARD_MOUNT_0=[-0.002,0.0,0.0]
ENV AIC_RSLRL_PUBLIC_SCRIPTED_TRACE_DELTA_NIC_CARD_MOUNT_1=[-0.002,0.0,0.0]
ENV AIC_RSLRL_ENABLE_SFP_GUARDED_INSERT=true
ENV AIC_RSLRL_SFP_GUARDED_INSERT_TARGETS=nic_card_mount_0,nic_card_mount_1
ENV AIC_RSLRL_SFP_GUARDED_INSERT_SEC=0.80
ENV AIC_RSLRL_SFP_GUARDED_INSERT_DOWN_STEP=-0.0007
ENV AIC_RSLRL_SFP_GUARDED_INSERT_LATERAL_STEP=0.0000
ENV AIC_RSLRL_SFP_GUARDED_INSERT_RETRACT_STEP=0.0015
ENV AIC_RSLRL_SFP_GUARDED_INSERT_FORCE_LIMIT=10.0
ENV AIC_RSLRL_SFP_GUARDED_INSERT_LINEAR_STIFFNESS=18.0
ENV AIC_RSLRL_SFP_GUARDED_INSERT_LINEAR_DAMPING=12.0
ENV AIC_RSLRL_ENABLE_SFP_TERMINAL_TARGET=true
ENV AIC_RSLRL_SFP_TERMINAL_TARGET_DWELL_SEC=1.20
ENV AIC_RSLRL_SFP_TERMINAL_TARGETS_NIC_CARD_MOUNT_0=[[-0.3848,0.2131,0.1904]]
ENV AIC_RSLRL_SFP_TERMINAL_TARGETS_NIC_CARD_MOUNT_1=[[-0.3848,0.2516,0.1874]]
ENV AIC_RSLRL_ENABLE_SFP_TCP_Z_RECOVERY=false
ENV AIC_RSLRL_ENABLE_SFP_TERMINAL_ORIENTATION=false
ENV AIC_RSLRL_ENABLE_SFP_LOCAL_SEARCH=false
ENV AIC_RSLRL_ENABLE_SFP_TERMINAL_FALLBACK_TRACE_SUFFIX=false
ENV AIC_RSLRL_ENABLE_SFP_TERMINAL_FALLBACK_JOINTS=false
ENV AIC_RSLRL_ENABLE_SC_TERMINAL_TARGET=true
ENV AIC_RSLRL_SC_TERMINAL_TARGET_DWELL_SEC=0.20
ENV AIC_RSLRL_SC_TERMINAL_TARGETS_SC_PORT_2=[[-0.4888,0.2838,0.0350]]
ENV AIC_RSLRL_ENABLE_SC_GUARDED_INSERT=true
ENV AIC_RSLRL_SC_GUARDED_INSERT_SEC=1.00
ENV AIC_RSLRL_SC_GUARDED_INSERT_DOWN_STEP=-0.0006
ENV AIC_RSLRL_SC_GUARDED_INSERT_RETRACT_STEP=0.0012
ENV AIC_RSLRL_SC_GUARDED_INSERT_FORCE_LIMIT=12.0
ENV AIC_RSLRL_SC_GUARDED_INSERT_LINEAR_STIFFNESS=15.0
ENV AIC_RSLRL_SC_GUARDED_INSERT_LINEAR_DAMPING=12.0
ENV AIC_RSLRL_ENABLE_SC_GUARDED_INSERT_JOINT_HOLD=false
ENV AIC_RSLRL_ENABLE_SC_TCP_Z_RECOVERY=false

RUN cd /ws_aic/src/aic && pixi install --frozen

WORKDIR /ws_aic/src/aic
COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

# prepare the credentials file
if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# start the model
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"
exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_model.RslRlCheckpointPolicy", "-p", "use_sim_time:=true"]
