import { ArrowLeftOutlined, CloseOutlined } from '@ant-design/icons';
import { Breadcrumb, Button, Card, Descriptions, Drawer, Image, Tag, Typography } from 'antd';
import moment from 'moment';
import Animate from 'rc-animate';
import React, { useEffect, useState } from 'react';
import { useModel, history } from 'umi';

import { Tabs } from 'antd';
import type { TabsProps } from 'antd';
import TableBase from '@/components/Table';
import FormIntent from './FormIntent';

import QuanLyCauHoi from '@/pages/QuanLyCauHoi';
import QuanLyCauTraLoi from '@/pages/QuanLyCauTraLoi';

const { Paragraph, Text } = Typography;
const { TabPane } = Tabs;
import './index.less';
moment.locale('vi');

const Device = ({ }) => {
  // const { socket, initSocket } = useModel('socket');
  const intentModel = useModel('intent');
  const name = intentModel?.record?.name ?? '';
  const recordDevice = intentModel?.record ?? {};
  const pathname = window.location.pathname;
  const topicID = pathname.split('/')[2];
  console.log(topicID, 'topicID');
  const IntentID = pathname.split('/')[4];

  const onClose = () => {
    intentModel.setVisibleForm(false);
  };

  const handleEdit = () => {
    intentModel.setEdit(true);
    intentModel.setVisibleForm(true);
  };

  useEffect(() => {
    if (!intentModel?.record?.id) intentModel.getDataById(IntentID);
  }, []);

  // console.log(intentModel?.listTrackIp, 'socket');
  return (
    <div>
      <Card>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Breadcrumb>
            <Breadcrumb.Item
              onClick={() => {
                history.push(`/topic/${topicID}/intent`);
              }}
            >
              <b style={{ cursor: 'pointer' }}>
                <ArrowLeftOutlined />
                Quay lại
              </b>
            </Breadcrumb.Item>
            <Breadcrumb.Item>
              <b>Thông tin: {name}</b>
            </Breadcrumb.Item>
          </Breadcrumb>
          <Button type="primary" htmlType="button" onClick={handleEdit}>
            Chỉnh sửa thông tin
          </Button>
        </div>
        <br />

        {/* <QuanLyCauTraLoi /> */}
        <Card title={`Thông tin ${name}`} style={{}}>
          <pre style={{ height: '400px', width: "100%", wordWrap: "break-word", overflowX: "unset" }}>{intentModel?.record?.file}</pre>
        </Card>
        <br />
        <QuanLyCauHoi />
        <Drawer title="Cập nhập" width={720} onClose={onClose} open={intentModel.visibleForm}>
          <FormIntent />
          <CloseOutlined
            onClick={() => {
              intentModel.setVisibleForm(false);
            }}
            style={{ position: 'absolute', top: 24, right: 24, cursor: 'pointer' }}
          />
        </Drawer>
      </Card>
    </div>
  );
};

export default Device;
