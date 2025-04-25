/* eslint-disable no-underscore-dangle */
import { IIntentRecord } from '@/models/intent';
import rules from '@/utils/rules';
import { Form, Input, Button, Card, Spin, DatePicker, InputNumber, Col, Row, Select } from 'antd';
import React, { useEffect, useState } from 'react';
import { useModel } from 'umi';

const layout = {
  labelCol: { span: 6 },
  wrapperCol: { span: 16 },
};
const tailLayout = {
  wrapperCol: { offset: 8, span: 16 },
};

const FormIntent = () => {
  const [form] = Form.useForm();
  const [status, setStatus] = useState<boolean>(false);
  const intentModel = useModel('intent');

  const handleFinish = async (values: IIntentRecord) => {
    if (intentModel.edit) {
      if (intentModel?.record?.id) {
        const data = document.getElementById('formIntent') as HTMLFormElement;
        const formData = new FormData(data);
        formData.append('intent_id', intentModel?.record?.id ?? '');
        await intentModel.upd({ id: intentModel?.record?.id, data: formData });
        // await intentModel.upd({
        //   ...intentModel?.record,
        //   ...values,
        //   id: intentModel?.record?.id,
        // });
        intentModel.setVisibleForm(false);
        intentModel.setEdit(false);
        intentModel.getData();
        intentModel.getDataById(intentModel?.record?.id ?? '');
      } else {
        const data = document.getElementById('formIntent') as HTMLFormElement;
        const formData = new FormData(data);
        await intentModel.add(formData);
      }
    } else {
      const data = document.getElementById('formIntent') as HTMLFormElement;
      const formData = new FormData(data);
      await intentModel.add(formData);
      intentModel.setVisibleForm(false);
      intentModel.setEdit(false);
    }
  };
  console.log(form.getFieldsValue(), 'form.getFieldValue()');
  return (
    <Spin spinning={intentModel.loading}>
      <Card title={intentModel.edit ? 'Chỉnh sửa' : 'Thêm mới'}>
        <Form
          id="formIntent"
          {...layout}
          form={form}
          onFinish={handleFinish}
          initialValues={{
            ...(intentModel?.record ?? {}),
          }}
        >
          <Form.Item label="Mã chủ đề" name="intent_name">
            <Input placeholder="Mã chủ đề" name="intent_name" />
          </Form.Item>
          <Form.Item label="Tên chủ đề" name="intent_detail">
            <Input placeholder="Tên chủ đề" name="intent_detail" />
          </Form.Item>
          <Form.Item {...tailLayout}>
            <Button type="primary" htmlType="submit">
              {intentModel.edit ? 'Cập nhật' : 'Thêm mới'}
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </Spin>
  );
};

export default FormIntent;
