/* eslint-disable radix */
/* eslint-disable no-underscore-dangle */
import TableBase from '@/components/Table';
import type { IIntentRecord } from '@/models/intent';
import { IColumn } from '@/utils/interfaces';
import { ArrowLeftOutlined, DeleteOutlined, EditOutlined, EyeOutlined } from '@ant-design/icons';
import { Breadcrumb, Button, Card, Divider, Popconfirm } from 'antd';
import React, { useEffect } from 'react';
import { useModel, history } from 'umi';
import FormIntent from './FormIntent';
import { ITopicRecord } from '@/models/outline';


const Index = () => {
  const pathname = window.location.pathname;
  const topicID = pathname.split('/')[2];
  console.log(topicID, 'topicID');
  const intentModel = useModel('intent');
  const name = intentModel?.record?.name ?? '';
  useEffect(() => {
    if (topicID) intentModel.getData(topicID);
  }, []);
  const handleEdit = (record: IIntentRecord) => {
    intentModel.setVisibleForm(true);
    intentModel.setEdit(true);
    intentModel.setRecord(record);
  };

  const handleDel = async (record: ITopicRecord) => {
    console.log("record delete", record);

    await intentModel.del(record?.id ?? "").then(() => {
      if (topicID) intentModel.getData(topicID);

    });
  };

  const renderLast = (value: any, record: IIntentRecord) => (
    <React.Fragment>
      <Button
        type="primary"
        shape="circle"
        icon={<EyeOutlined />}
        title="Xem chi tiết"
        onClick={() => {
          console.log(record, 'record');
          intentModel.setRecord(record);
          history.push(`/topic/${topicID}/intent/${record.id}`);
        }}
      />
      <Divider type="vertical" />
      <Button
        type="primary"
        shape="circle"
        icon={<EditOutlined />}
        title="Chỉnh sửa"
        onClick={() => {
          console.log(record, 'record');
          handleEdit(record);
        }}
      />
      <Divider type="vertical" />
      <Popconfirm
        title="Bạn có muốn xóa?"
        okText="Có"
        cancelText="Không"
        onConfirm={() => handleDel(record)}
      >
        <Button type="danger" shape="circle" icon={<DeleteOutlined />} title="Xóa" />
      </Popconfirm>
    </React.Fragment>
  );
  const columns: IColumn<IIntentRecord>[] = [
    {
      title: 'STT',
      dataIndex: 'index',
      width: 80,
      align: 'center',
    },
    // {
    //   title: 'Mã chủ đề',
    //   dataIndex: 'intent_name',
    //   notRegex: true,
    //   width: 200,
    //   align: 'left',
    //   // render: (value: any, record: IIntentRecord) => (
    //   //   <img src={record.avatar} alt={record.name} style={{ width: 100, height: 100 }} />
    //   // ),
    // },
    {
      title: 'Tên thông tin',
      dataIndex: 'outline_detail',
      search: 'search',
      notRegex: true,
      width: 200,
      align: 'left',
      render: (value: any, record: IIntentRecord) => (
        record?.name
      ),
    },
    // {
    //   title: 'Trạng thái',
    //   dataIndex: 'status',
    //   render: (value: any, record: IIntentRecord) => (
    //     <span>{record.status === true ? 'Hoạt động' : 'Không hoạt động'}</span>
    //   ),
    //   search: 'search',
    //   notRegex: true,
    //   width: 200,
    //   align: 'center',
    // },
    {
      title: 'Thao tác',
      align: 'center',
      render: (value: any, record: IIntentRecord) => renderLast(value, record),
      fixed: 'right',
      width: 200,
    },
  ];

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
                history.push(`/topic`);
              }}
            >
              <b style={{ cursor: 'pointer' }}>
                <ArrowLeftOutlined />
                Quay lại
              </b>
            </Breadcrumb.Item>
            {/* <Breadcrumb.Item>
              <b>Thông tin: {name}</b>
            </Breadcrumb.Item> */}
          </Breadcrumb>
        </div>
        <br />
        <TableBase
          modelName={'intent'}
          title="Danh sách thông tin trong chủ đề"
          columns={columns}
          hascreate={true}
          formType={'Modal'}
          dependencies={[intentModel.page, intentModel.limit, intentModel.condition]}
          widthDrawer={800}
          getData={() => { if (topicID) intentModel.getData(topicID) }}
          Form={FormIntent}
          noCleanUp={true}
          params={{
            page: intentModel.page,
            size: intentModel.limit,
            condition: intentModel.condition,
          }}
          maskCloseableForm={true}
          otherProps={{
            scroll: {
              x: 1000,
            },
          }}
        />
      </Card>
    </div>
  );
};

export default Index;
