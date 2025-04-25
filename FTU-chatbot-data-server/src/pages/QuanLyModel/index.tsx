/* eslint-disable radix */
/* eslint-disable no-underscore-dangle */
import TableBase from '@/components/Table';
import type { IModelRecord } from '@/models/model';
import { IColumn } from '@/utils/interfaces';
import { DeleteOutlined, EditOutlined, EyeOutlined, PoweroffOutlined } from '@ant-design/icons';
import { Button, Card, Divider, Popconfirm, Progress } from 'antd';
import React from 'react';
import { useModel, history } from 'umi';
import FormModel from './FormModel';

const Index = () => {
  const modelModel = useModel('model');

  const handleEdit = async (record: IModelRecord) => {
    const formData = new FormData();
    formData.append('model_name', record?.model_name ?? '');
    await modelModel.activate(formData).then(() => {
      modelModel.getData();
    });
  };

  const handleDel = async (record: IModelRecord) => {
    await modelModel.del(record?.id ?? '').then(() => {
      modelModel.getData();
    });
  };

  const renderLast = (value: any, record: IModelRecord) => (
    <React.Fragment>
      {/* <Button
        type="primary"
        shape="circle"
        icon={<EyeOutlined />}
        title="Xem chi tiết"
        onClick={() => {
          console.log(record, 'record');
          modelModel.setRecord(record);
          history.push(`/model/${record.id}`);
        }}
      />
      <Divider type="vertical" />*/}
      <Popconfirm
        title="Bạn có sử dụng mô hình này?"
        okText="Có"
        cancelText="Không"
        onConfirm={() => handleEdit(record)}
      >
        <Button type="primary" shape="circle" icon={<PoweroffOutlined />} title="Sử dụng" />
      </Popconfirm>
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
  const columns: IColumn<IModelRecord>[] = [
    {
      title: 'STT',
      dataIndex: 'index',
      width: 80,
      align: 'center',
    },
    {
      title: 'Tên mô hình',
      dataIndex: 'model_name',
      notRegex: true,
      width: 200,
      search: 'sort',
      align: 'left',
      // render: (value: any, record: IModelRecord) => (
      //   <img src={record.avatar} alt={record.name} style={{ width: 100, height: 100 }} />
      // ),
    },
    // {
    //   title: 'Tên chủ đề',
    //   dataIndex: 'model_detail',
    //   search: 'search',
    //   notRegex: true,
    //   width: 200,
    //   align: 'left',
    // },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      render: (value: any, record: IModelRecord) => (
        <span
          style={{
            color: record.isActivate === true ? 'green' : 'red',
          }}
        >
          {record.isActivate === true ? 'Hoạt động' : 'Không hoạt động'}
        </span>
      ),
      search: 'sort',
      notRegex: true,
      width: 120,
      align: 'center',
    },
    {
      title: 'Ngày tạo',
      dataIndex: 'created_at',
      search: 'search',
      notRegex: true,
      width: 120,
      align: 'center',
      render: (value: any, record: IModelRecord) => (
        <span>
          {new Date(record.created_at).toLocaleDateString('vi-VN', {
            timeZone: 'UTC',
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      ),
    },
    {
      title: 'Thao tác',
      align: 'center',
      render: (value: any, record: IModelRecord) => renderLast(value, record),
      fixed: 'right',
      width: 200,
    },
  ];

  return (
    <>
      <Card>
        <Button
          type="primary"
          loading={modelModel.loadingTrain}
          onClick={() => {
            modelModel.autoTrain();
          }}
        >
          {modelModel.loadingTrain ? 'Đang cập nhập dữ liệu mới' : 'Cập nhập dữ liệu mới'}
        </Button>
        {/* {modelModel.isClickTrain && (
          <>
            <Progress
              percent={modelModel.percent}
              status={modelModel.status}
              style={{ marginTop: 10 }}
            />
          </>
        )} */}
      </Card>
      {/* <TableBase
        modelName={'model'}
        title="Các mô hình đã tạo"
        columns={columns}
        hascreate={false}
        formType={'Modal'}
        dependencies={[modelModel.page, modelModel.limit, modelModel.condition]}
        widthDrawer={800}
        getData={modelModel.getData}
        Form={FormModel}
        noCleanUp={true}
        params={{
          page: modelModel.page,
          size: modelModel.limit,
          condition: modelModel.condition,
        }}
        maskCloseableForm={true}
        otherProps={{
          scroll: {
            x: 1000,
          },
        }}
      /> */}
    </>
  );
};

export default Index;
