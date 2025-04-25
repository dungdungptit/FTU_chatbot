/* eslint-disable radix */
/* eslint-disable no-underscore-dangle */
import TableBase from '@/components/Table';
import type { IIntentRecord, ITopicRecord } from '@/models/outline';
import { IColumn } from '@/utils/interfaces';
import { DeleteOutlined, EditOutlined, EyeOutlined } from '@ant-design/icons';
import { Button, Card, Divider, Popconfirm } from 'antd';
import React from 'react';
import { useModel, history } from 'umi';
import FormIntent from './FormIntent';

const Index = () => {
  const outlineModel = useModel('outline');

  const handleEdit = (record: IIntentRecord) => {
    outlineModel.setEdit(true);
    outlineModel.setRecord(record);
    outlineModel.setVisibleForm(true);
  };

  const handleDel = async (record: IIntentRecord) => {
    await outlineModel.del(record?.id ?? '').then(() => {
      outlineModel.getData();
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
          outlineModel.setRecord(record);
          history.push(`/topic/${record.id}/intent`);
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
    //   dataIndex: 'outline_name',
    //   notRegex: true,
    //   width: 200,
    //   align: 'left',
    //   // render: (value: any, record: IIntentRecord) => (
    //   //   <img src={record.avatar} alt={record.name} style={{ width: 100, height: 100 }} />
    //   // ),
    // },
    {
      title: 'Tên chủ đề',
      dataIndex: 'outline_detail',
      search: 'search',
      notRegex: true,
      width: 200,
      align: 'left',
      render: (value: any, record: ITopicRecord) => (
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
    <>
      <TableBase
        modelName={'outline'}
        title="Danh sách chủ đề"
        columns={columns}
        hascreate={true}
        formType={'Modal'}
        dependencies={[outlineModel.page, outlineModel.limit, outlineModel.condition]}
        widthDrawer={800}
        getData={outlineModel.getData}
        Form={FormIntent}
        noCleanUp={true}
        total={outlineModel.total}
        params={{
          page: outlineModel.page,
          size: outlineModel.limit,
          condition: outlineModel.condition,
        }}
        maskCloseableForm={true}
        otherProps={{
          scroll: {
            x: 1000,
          },
        }}
      />
    </>
  );
};

export default Index;
